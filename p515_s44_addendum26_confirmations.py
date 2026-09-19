"""P5.15 Addendum 26 -- zero-solve confirmations (frozen spec v14, key
`addendum26_confirmations_zero_solve`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 26; STEP4_DFO_METHOD.md v1 Sec 1.2-1.3
and Sec 9; data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json.

ZERO SOLVES. A `SolveProfileGuard` with NO permitted call site is installed before any
production module is imported and stays installed for the whole run; `verify(0)` is
checked exactly (zero permitted solves, zero process launches, zero blocked attempts).

  1. Cost-file provenance of data/SRP1/SharedESS/SRP1_ESS.xlsx: git history on HEAD
     and on EVERY local/remote ref, the distinct blobs of that path across refs, a
     cell-by-cell comparison of those blobs, working copy vs HEAD, sha256, the
     unit-cost sheets as production reads them, and a text search of the repository
     documents for anything naming "the corrected file".
  2. I(x) = `model.investment_cost` of the production Benders master
     (`shared_energy_storage_data._build_master_problem`), BUILT and EVALUATED at each
     candidate (Var values loaded with production's own
     `load_candidate_solution_into_master_model`; no solve), cross-checked against
     the independent transcription `p56a_oracle.investment_cost`. Budget slack read
     from the master's OWN budget row (ub - body). Evaluated with the committed cost
     file and, as a labelled sensitivity, with every other distinct blob of the same
     path found in the repository (read by production's own reader from a temporary
     copy). Where `budget` / `max_capacity` are read in production (AST scan) and
     which call sites reach those readers.
  3. x = 0 at EVERY node: the oracle's construction path
     (`p515_g_g1_g4_admm_gates._construct_arm_planning`, the D arm's `apply_rho=False`)
     then production's own initialization sequence of `_run_operational_planning`
     (create_admm_variables, create_distribution_networks_models,
     create_transmission_network_model, create_shared_energy_storage_model, the ADMM
     objective preparation, S_ref normalization, consensus initialization, capacities
     publication), one ESSO coordination update, residual metrics and every evaluator
     capture -- with each agent's `.optimize` replaced, on that planning instance
     only, by an INTERCEPTOR that records the call and returns "no result" (never a
     solver, never a fabricated solution). The declared intercept count is checked
     exactly. Plus a read-only scan of preserved evidence for zero-capacity runs.
  4. ESSO capacity multipliers: the rows linking ESSO operation to installed s/e
     (located in source), the Suffix declaration and the post-solve load, and
     read-only inspection (unpickling, no solve) of the committed ESSO models
     `esso_models_s39_D.pkl`, the hash-recorded `certified_models.pkl` and the
     `esso_capture/` cycle-139 records of the S42 exact-fix re-run.

Usage (attached, both streams captured by the caller):
    python p515_s44_addendum26_confirmations.py            # the run (write-once)
    python p515_s44_addendum26_confirmations.py --manifest  # sha256 manifest of the dir
"""
import ast
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = 'P5.15 Addendum 26 zero-solve confirmations (frozen spec v14)'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 26',
    'STEP4_DFO_METHOD.md v1 Sec 1.2-1.3, Sec 9',
    'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json '
    '(key addendum26_confirmations_zero_solve)',
]
SPEC_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44',
                         'frozen_s44_selection_spec_v14_e4500e27.json')
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'confirmations')
XLSX_REL = 'data/SRP1/SharedESS/SRP1_ESS.xlsx'
XLSX_PATH = os.path.join(REPO, XLSX_REL)
S42_DIR_REL = 'data/SRP1/Results/P515S42/exact_fix_rerun'

OUTPUT_FILES = (
    'item1_cost_file_provenance.json',
    'item2_investment_cost_and_budget_slack.json',
    'item3_zero_investment_evaluability.json',
    'item4_esso_capacity_multipliers.json',
    'production_stdout_capture.log',
)
MANIFEST_NAME = 'manifest_sha256.json'

# Candidates. Year 2025 for every candidate (frozen spec v14 item1_harness
# `candidate_definition`: "investment year 2025 as for C*"). Sources cited per entry.
INVEST_YEAR = 2025
CANDIDATES = {
    'paper_plan': {
        'source': 'spec v14 item3_selection_run.candidates.paper_plan; Addendum 26 task',
        'map': {7: (1.62, 3.24)}},
    'c_star': {
        'source': 'spec v14 item3_selection_run.candidates.C_star; Addendum 26 task',
        'map': {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}},
    'lattice_c_star': {
        'source': 'Addendum 26 "Reference candidates on the lattice": C* -> 1.0 MVA / 4.0 MWh '
                  'at all nodes',
        'map': {5: (1.0, 4.0), 7: (1.0, 4.0), 9: (1.0, 4.0)}},
    'lattice_paper_plan': {
        'source': 'Addendum 26 "Reference candidates on the lattice": plan -> 1.5 MVA / 3.0 MWh '
                  'at node 7',
        'map': {7: (1.5, 3.0)}},
    'node7_empty': {
        'source': 'spec v14 item3_selection_run.candidates.node7_empty (context only)',
        'map': {5: (0.96875, 3.875), 9: (0.96875, 3.875)}},
    'two_c_star': {
        'source': 'spec v14 item3_selection_run.candidates.two_c_star (context only)',
        'map': {5: (1.9375, 7.75), 7: (1.9375, 7.75), 9: (1.9375, 7.75)}},
    'zero': {
        'source': 'item 3 (x = 0 at every node)',
        'map': {}},
}
# STEP4_DFO_METHOD.md Sec 1.2, quoted verbatim for comparison only (not used in any
# computation): "At the file's 2025 expected unit costs (~ EUR 256k/MVA, ~ EUR 203k/MWh)".
STEP4_QUOTED_2025_UNIT_COSTS = {'power_eur_per_mva_approx': 256e3,
                                'energy_eur_per_mwh_approx': 203e3,
                                'source': 'STEP4_DFO_METHOD.md Sec 1.2 (quoted, rounded)'}

# Item 3: declared BEFORE the run -- the exact number of `.optimize` interceptions the
# traced x = 0 sequence must produce: 3 DSO (one per node, sequential creation), 1 TSO,
# 1 ESSO initialization, 1 ESSO coordination update.
DECLARED_INTERCEPTS = {'dso': 3, 'tso': 1, 'esso_init': 1, 'esso_coordination': 1}


# ======================================================================================
#  small utilities
# ======================================================================================
def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, binary=False):
    res = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=not binary)
    return res.returncode, res.stdout, res.stderr


def _find_line(rel_path, needle, occurrence=1):
    """1-based line number of the `occurrence`-th line containing `needle`; raises if
    absent, so every file:line this script reports is computed, never typed."""
    count = 0
    with open(os.path.join(REPO, rel_path)) as fh:
        for lineno, line in enumerate(fh, start=1):
            if needle in line:
                count += 1
                if count == occurrence:
                    return lineno
    raise LookupError(f'{needle!r} (occurrence {occurrence}) not found in {rel_path}')


def _loc(rel_path, needle, occurrence=1):
    return f'{rel_path}:{_find_line(rel_path, needle, occurrence)}'


def _func_span(func):
    src, start = inspect.getsourcelines(func)
    return f'{os.path.relpath(inspect.getsourcefile(func), REPO)}:{start}-{start + len(src) - 1}'


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):
        return 'NaN'
    return obj


def _preflight_no_concurrent_harness():
    """Read-only: refuse to run while another p5* harness process is alive (this run
    reads the shared baseline and would contend for the machine). Never touches lock
    files or processes."""
    res = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    me = os.getpid()
    others = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid == me or 'p515_s44_addendum26_confirmations' in cmd:
            continue
        if 'python' in cmd and re.search(r'\bp5\d', cmd):
            others.append(line.strip())
    if others:
        raise RuntimeError('refusing to run concurrently with another harness process: '
                           + ' | '.join(others))
    return {'checked_with': 'ps -axo pid=,command=', 'other_p5_harness_processes': []}


# ======================================================================================
#  Item 1 -- cost-file provenance
# ======================================================================================
def _read_workbook_bytes(data):
    import openpyxl
    import pandas as pd
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        wb = openpyxl.load_workbook(path, data_only=True)
        props = wb.properties
        core = {'created': str(props.created), 'modified': str(props.modified),
                'creator': props.creator, 'last_modified_by': props.lastModifiedBy}
    finally:
        os.unlink(path)
    out = {}
    for name, df in sheets.items():
        out[name] = [[(None if (isinstance(v, float) and v != v) else
                       (v.item() if hasattr(v, 'item') else v)) for v in row]
                     for row in df.itertuples(index=False, name=None)]
    return out, core


def _compare_workbooks(a, b):
    """Cell-by-cell comparison of two parsed workbooks (sheet -> rows)."""
    report = {}
    for sheet in sorted(set(a) | set(b)):
        ra, rb = a.get(sheet), b.get(sheet)
        if ra is None or rb is None:
            report[sheet] = {'present_in_both': False}
            continue
        diffs, ratios = [], []
        for i in range(max(len(ra), len(rb))):
            row_a = ra[i] if i < len(ra) else []
            row_b = rb[i] if i < len(rb) else []
            for j in range(max(len(row_a), len(row_b))):
                va = row_a[j] if j < len(row_a) else None
                vb = row_b[j] if j < len(row_b) else None
                if va != vb:
                    diffs.append({'row': i, 'col': j, 'a': va, 'b': vb})
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
                        ratios.append(vb / va)
        entry = {'present_in_both': True, 'n_differing_cells': len(diffs)}
        if diffs:
            entry['first_differences'] = diffs[:6]
        if ratios:
            entry['numeric_ratio_b_over_a'] = {'min': min(ratios), 'max': max(ratios),
                                               'n': len(ratios)}
        report[sheet] = entry
    return report


def _expected_unit_costs(sed):
    """Scenario-weighted, discounted unit costs per case year, from a SharedEnergyStorage-
    Data object's OWN loaded fields (the same quantities the master's expression uses)."""
    years = list(sed.years)
    out = {}
    for year in years:
        disc = 1.0 / ((1.0 + sed.discount_factor) ** (int(year) - int(years[0])))
        ps = sum(w * sed.cost_investment['power'][m][year]
                 for m, w in enumerate(sed.prob_market_scenarios))
        es = sum(w * sed.cost_investment['energy'][m][year]
                 for m, w in enumerate(sed.prob_market_scenarios))
        out[str(year)] = {'discount_multiplier': disc,
                          'power_eur_per_mva_undiscounted': ps,
                          'energy_eur_per_mwh_undiscounted': es,
                          'power_eur_per_mva_discounted': disc * ps,
                          'energy_eur_per_mwh_discounted': disc * es}
    return out


def _doc_search():
    """Scoped search of repository documents for anything identifying the corrected
    cost file. Records exactly what was searched."""
    tokens = ['SRP1_ESS', 'Costs updated', '7ce1d1ab', '5ef61f5d', '8a7e3744',
              '14581474', 'e17bd588', 'cost file', 'Cost file']
    tracked_hits = []
    for tok in tokens:
        rc, out, _ = _git(['grep', '-n', '-I', '-F', tok, 'HEAD', '--',
                           '*.md', '*.py', '*.json', '*.txt'])
        for line in out.splitlines():
            if line.startswith('HEAD:p515_s44_addendum26_confirmations.py'):
                continue
            tracked_hits.append({'token': tok, 'hit': line[:300]})
    untracked_docs = ['PLANNER_BRIEF_2026-09-13.md', 'STEP4_DFO_METHOD.md', 'EXPERT_REVIEW.md',
                      'EXPERT_REVIEW_2_ACTION_PLAN.md', 'CLAUDE_LEGACY_BACKUP.md']
    untracked_hits = []
    for doc in untracked_docs:
        path = os.path.join(REPO, doc)
        if not os.path.exists(path):
            untracked_hits.append({'doc': doc, 'exists': False})
            continue
        with open(path, errors='replace') as fh:
            for lineno, line in enumerate(fh, start=1):
                for tok in tokens + ['corrected', 'investment cost file']:
                    if tok in line and ('cost' in line.lower() or tok in tokens):
                        untracked_hits.append({'doc': doc, 'line': lineno, 'token': tok,
                                               'text': line.strip()[:300]})
                        break
    return {
        'searched': {
            'tracked_files_at_HEAD': "git grep -n -I -F <token> HEAD -- '*.md' '*.py' "
                                     "'*.json' '*.txt' (this script excluded)",
            'documents_read_in_full_from_working_tree': untracked_docs,
            'tokens': tokens + ['corrected (only on lines mentioning cost)',
                                'investment cost file'],
        },
        'tracked_hits': tracked_hits,
        'untracked_document_hits': untracked_hits,
    }


def item1_cost_file_provenance(planning, SED):
    working_bytes = open(XLSX_PATH, 'rb').read()
    rc, head_blob, _ = _git(['rev-parse', f'HEAD:{XLSX_REL}'])
    head_blob = head_blob.strip()
    rc, head_bytes, _ = _git(['cat-file', 'blob', head_blob], binary=True)
    rc, porcelain, _ = _git(['status', '--porcelain', '--', XLSX_REL])
    rc, hash_object, _ = _git(['hash-object', XLSX_REL])

    # History reachable from HEAD, rename-tracked.
    rc, follow, _ = _git(['log', '--follow', '--name-status',
                          '--format=COMMIT|%H|%aI|%an|%s', '--', XLSX_REL])
    head_history = []
    for block in follow.split('COMMIT|')[1:]:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        commit, date, author, subject = lines[0].split('|', 3)
        head_history.append({'commit': commit, 'date': date, 'author': author,
                             'subject': subject, 'name_status': lines[1:]})

    # Every commit, on ANY ref, touching the exact current path.
    rc, all_log, _ = _git(['log', '--all', '--format=%H|%aI|%an|%s', '--', XLSX_REL])
    all_commits = []
    for line in all_log.strip().splitlines():
        commit, date, author, subject = line.split('|', 3)
        anc = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
                             cwd=REPO).returncode == 0
        _, blob_after, _ = _git(['rev-parse', f'{commit}:{XLSX_REL}'])
        rcp, blob_before, _ = _git(['rev-parse', f'{commit}^:{XLSX_REL}'])
        _, branches, _ = _git(['branch', '-a', '--contains', commit])
        all_commits.append({
            'commit': commit, 'date': date, 'author': author, 'subject': subject,
            'is_ancestor_of_HEAD': anc,
            'blob_after': blob_after.strip(),
            'blob_before': blob_before.strip() if rcp == 0 else None,
            'refs_containing': [b.strip().lstrip('* ').strip() for b in branches.splitlines()],
        })

    # Distinct blobs of the path across every local and remote ref.
    rc, refs, _ = _git(['for-each-ref', '--format=%(refname:short)', 'refs/heads',
                        'refs/remotes'])
    blob_to_refs, refs_without_path = {}, []
    for ref in refs.split():
        rcb, blob, _ = _git(['rev-parse', '-q', '--verify', f'{ref}:{XLSX_REL}'])
        if rcb != 0 or not blob.strip():
            refs_without_path.append(ref)
            continue
        blob_to_refs.setdefault(blob.strip(), []).append(ref)

    blobs = {}
    parsed = {}
    for blob, refs_list in blob_to_refs.items():
        _, data, _ = _git(['cat-file', 'blob', blob], binary=True)
        sheets, core = _read_workbook_bytes(data)
        parsed[blob] = (data, sheets)
        blobs[blob] = {'sha256': _sha256_bytes(data), 'bytes': len(data),
                       'n_refs': len(refs_list),
                       'is_HEAD_blob': blob == head_blob,
                       'refs': sorted(refs_list),
                       'xlsx_core_properties': core}
    comparisons = {}
    for blob in parsed:
        if blob == head_blob:
            continue
        comparisons[f'HEAD({head_blob[:8]}) vs {blob[:8]}'] = _compare_workbooks(
            parsed[head_blob][1], parsed[blob][1])

    mb = None
    if any(r == 'paper_revisions' for rl in blob_to_refs.values() for r in rl):
        _, mb_out, _ = _git(['merge-base', 'HEAD', 'paper_revisions'])
        mb = mb_out.strip()
        _, mb_info, _ = _git(['log', '-1', '--format=%H|%aI|%s', mb])
        rc_mb, mb_blob, _ = _git(['rev-parse', f'{mb}:{XLSX_REL}'])
        mb = {'merge_base_HEAD_paper_revisions': mb_info.strip(),
              'blob_at_merge_base': mb_blob.strip() if rc_mb == 0 else None}

    sed = planning.shared_ess_data
    working_sheets, working_core = _read_workbook_bytes(working_bytes)
    unit_cost_sheets = {k: working_sheets[k] for k in
                        ('Scenarios', 'Investment Cost, Power', 'Investment Cost, Energy')}

    content_changing = [c for c in all_commits
                        if c['blob_before'] is not None and c['blob_after'] != c['blob_before']]
    not_on_head = [c for c in content_changing if not c['is_ancestor_of_HEAD']]
    newer_blobs_not_on_head = sorted({c['blob_after'] for c in not_on_head})
    head_blob_is_parent_of_those = all(c['blob_before'] == head_blob for c in not_on_head)

    verdict = {
        'working_copy_sha256': _sha256_bytes(working_bytes),
        'working_copy_matches_HEAD_blob_bytes': working_bytes == head_bytes,
        'git_status_porcelain_clean': porcelain.strip() == '',
        'HEAD_blob': head_blob,
        'n_distinct_blobs_across_all_refs': len(blob_to_refs),
        'content_changing_commits_at_this_exact_path_on_any_ref': [
            {k: c[k] for k in ('commit', 'date', 'subject', 'is_ancestor_of_HEAD',
                               'blob_before', 'blob_after')} for c in content_changing],
        'content_changing_commits_NOT_on_HEAD': [
            {k: c[k] for k in ('commit', 'date', 'subject', 'refs_containing')}
            for c in not_on_head],
        'blobs_introduced_by_commits_not_on_HEAD': newer_blobs_not_on_head,
        'HEAD_blob_is_the_parent_version_of_every_such_commit': head_blob_is_parent_of_those,
        'statement': None,
    }
    if not_on_head:
        verdict['statement'] = (
            f'The committed file on this branch (blob {head_blob[:8]}, sha256 '
            f'{_sha256_bytes(head_bytes)[:16]}...) is byte-identical to the working copy. The '
            f'repository holds {len(not_on_head)} content-changing commit(s) to this exact path '
            'that are NOT ancestors of HEAD: '
            + '; '.join(f"{c['commit'][:8]} {c['date']} '{c['subject']}' on "
                        f"{', '.join(c['refs_containing'])}" for c in not_on_head)
            + f'. HEAD carries the version those commits replaced (parent blob == HEAD blob: '
            f'{head_blob_is_parent_of_those}). Therefore the committed file on this branch is '
            'NOT the latest cost update present in the repository. Whether that update is the '
            '"correction" Addendum 26 refers to cannot be established from the repository: '
            'no document found by the recorded search names a commit, blob or hash for the '
            'corrected file.')
    else:
        verdict['statement'] = (
            'No content-changing commit to this path exists outside HEAD\'s history; the '
            'committed file is the latest version present in the repository.')

    return {
        'path': XLSX_REL,
        'working_copy': {'sha256': _sha256_bytes(working_bytes), 'bytes': len(working_bytes),
                         'git_hash_object': hash_object.strip(),
                         'xlsx_core_properties': working_core},
        'HEAD_blob': {'blob': head_blob, 'sha256': _sha256_bytes(head_bytes)},
        'git_status_porcelain': porcelain.strip(),
        'history_reachable_from_HEAD_follow': head_history,
        'commits_on_any_ref_touching_exact_path': all_commits,
        'distinct_blobs_across_refs': blobs,
        'refs_without_this_path': refs_without_path,
        'merge_base': mb,
        'blob_comparisons_cell_by_cell': comparisons,
        'unit_cost_sheets_working_copy_raw': unit_cost_sheets,
        'as_read_by_production': {
            'reader': _func_span(SED._read_shared_energy_storage_data_from_file),
            'scenario_weights_omega_m': list(sed.prob_market_scenarios),
            'scenario_weights_stored_at': f'{XLSX_REL} sheet "Scenarios", row 1, columns C-E '
                                          '(read into shared_ess_data.prob_market_scenarios; '
                                          'the name is historical -- they are the '
                                          'investment-cost scenario weights of this workbook)',
            'cost_investment_power_eur_per_mva': sed.cost_investment['power'],
            'cost_investment_energy_eur_per_mwh': sed.cost_investment['energy'],
            'discount_factor': sed.discount_factor,
            'discount_factor_stored_at': f'data/SRP1/SRP1.json "DiscountFactor" '
                                         f'(line {_find_line("data/SRP1/SRP1.json", "DiscountFactor")}), '
                                         'assigned at ' + _loc('shared_resources_planning.py',
                                                               'shared_ess_data.discount_factor = planning_problem.discount_factor')
                                         + '; NOT stored in the xlsx',
            'expected_unit_costs_per_case_year': _expected_unit_costs(sed),
        },
        'document_search': _doc_search(),
        'provenance_verdict': verdict,
    }


# ======================================================================================
#  Item 2 -- I(x) and budget slack
# ======================================================================================
def _full_candidate(O, planning, node_map):
    nodes, years = O.nodes_and_years(planning)
    x = {}
    for n in nodes:
        for y in years:
            s, e = node_map.get(n, (0.0, 0.0)) if y == INVEST_YEAR else (0.0, 0.0)
            x[(n, y)] = {'s': s, 'e': e}
    return O.vector_to_candidate(planning, x)


def _master_rows_state(model, pe):
    budget_row = model.energy_storage_investment[1]
    body = pe.value(budget_row.body)
    maxcap = []
    for idx, row in model.energy_storage_maximum_capacity.items():
        v = pe.value(row.body)
        if v > pe.value(row.upper) + 1e-9:
            maxcap.append({'row': idx, 'body_mwh': v, 'ub': pe.value(row.upper)})
    ratio = []
    for idx, row in model.energy_storage_power_to_energy_factor.items():
        v = pe.value(row.body)
        lo = pe.value(row.lower) if row.lower is not None else None
        up = pe.value(row.upper) if row.upper is not None else None
        if (lo is not None and v < lo - 1e-9) or (up is not None and v > up + 1e-9):
            ratio.append({'row': idx, 'body': v, 'lb': lo, 'ub': up})
    return {'budget_row_body_eur': body, 'budget_row_ub_eur': pe.value(budget_row.upper),
            'budget_slack_from_row_eur': pe.value(budget_row.upper) - body,
            'max_capacity_rows_violated': maxcap, 'ratio_rows_violated': ratio}


def _evaluate_candidates(O, srp, planning_like, sed, pe, label):
    master = sed.build_master_problem()
    nodes = list(sed.active_distribution_network_nodes)
    out = {}
    for name, spec in CANDIDATES.items():
        cand = _full_candidate(O, planning_like['planning_for_candidates'], spec['map'])
        sed.load_candidate_solution_into_master_model(master, cand)
        i_master = pe.value(master.investment_cost)
        rows = _master_rows_state(master, pe)
        per_node = {}
        for n in nodes:
            c_n = _full_candidate(O, planning_like['planning_for_candidates'],
                                  {n: spec['map'][n]} if n in spec['map'] else {})
            sed.load_candidate_solution_into_master_model(master, c_n)
            per_node[str(n)] = pe.value(master.investment_cost)
        feasible, reason = srp._check_candidate_first_stage_feasibility(
            planning_like['planning_for_feasibility'], cand)
        entry = {
            'source': spec['source'],
            'investment_mva_mwh_year_2025': {str(k): {'s_mva': v[0], 'e_mwh': v[1]}
                                             for k, v in spec['map'].items()},
            'I_x_eur_master_expression': i_master,
            'I_x_per_node_eur': per_node,
            'budget_eur': sed.params.budget,
            'budget_slack_B_minus_I_eur': sed.params.budget - i_master,
            'I_over_B': i_master / sed.params.budget,
            'master_rows': rows,
            'first_stage_feasible': feasible,
            'first_stage_reasons': reason,
        }
        if label == 'committed':
            entry['I_x_eur_p56a_transcription'] = O.investment_cost(
                planning_like['planning_for_candidates'], cand)
            entry['abs_diff_master_vs_transcription_eur'] = abs(
                entry['I_x_eur_p56a_transcription'] - i_master)
        out[name] = entry
    return out


def _attribute_scan(attrs):
    """AST scan of production modules for attribute reads of `attrs`, with the
    enclosing function; plus call sites of the functions that read them."""
    rc, files, _ = _git(['ls-files', '*.py'])
    prod = sorted(f for f in files.split()
                  if '/' not in f and not re.match(r'^p\d', f)
                  and f not in ('audit_p3_snapshots.py', 'validate_vmag_refactor.py', 'debug.py'))
    reads = []
    for f in prod:
        tree = ast.parse(open(os.path.join(REPO, f)).read())
        spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in attrs:
                enclosing = [s for s in spans if s[0] <= node.lineno <= s[1]]
                fn = min(enclosing, key=lambda s: s[1] - s[0])[2] if enclosing else '<module>'
                reads.append({'file': f, 'line': node.lineno, 'attr': node.attr,
                              'ctx': type(node.ctx).__name__, 'function': fn})
    readers = sorted({r['function'] for r in reads})
    calls = []
    for f in prod:
        tree = ast.parse(open(os.path.join(REPO, f)).read())
        spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else node.func.id if isinstance(node.func, ast.Name) else None)
                if name in readers or (name and name.lstrip('_') in {r.lstrip('_') for r in readers}):
                    enclosing = [s for s in spans if s[0] <= node.lineno <= s[1]]
                    fn = min(enclosing, key=lambda s: s[1] - s[0])[2] if enclosing else '<module>'
                    calls.append({'file': f, 'line': node.lineno, 'callee': name,
                                  'caller': fn})
    return {'production_files_scanned': prod, 'attribute_accesses': reads,
            'reader_functions': readers, 'call_sites_of_reader_functions': calls}


def item2_investment_cost(planning, O, srp, SED, G, N, pe, alt_blobs):
    from pyomo.core.expr.visitor import identify_variables
    import types
    sed = planning.shared_ess_data
    committed = _evaluate_candidates(
        O, srp, {'planning_for_candidates': planning, 'planning_for_feasibility': planning},
        sed, pe, 'committed')

    alternatives = {}
    for blob, data in alt_blobs.items():
        alt_sed = deepcopy(sed)
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            with redirect_stdout(io.StringIO()):
                SED._read_shared_energy_storage_data_from_file(alt_sed, tmp_path)
        finally:
            os.unlink(tmp_path)
        shim = types.SimpleNamespace(shared_ess_data=alt_sed)
        alternatives[blob] = {
            'blob_sha256': _sha256_bytes(data),
            'read_by': _func_span(SED._read_shared_energy_storage_data_from_file)
                       + ' (production reader, temporary copy of the git blob)',
            'expected_unit_costs_per_case_year': _expected_unit_costs(alt_sed),
            'candidates': _evaluate_candidates(
                O, srp, {'planning_for_candidates': planning,
                         'planning_for_feasibility': shim}, alt_sed, pe, 'alternative'),
        }

    master = sed.build_master_problem()
    maxcap_vars = sorted({v.parent_component().name for row in
                          master.energy_storage_maximum_capacity.values()
                          for v in identify_variables(row.body)})
    budget_vars = sorted({v.parent_component().name for v in
                          identify_variables(master.energy_storage_investment[1].body)})
    scan = _attribute_scan({'budget', 'max_capacity', 'min_energy_to_power_ratio',
                            'max_energy_to_power_ratio'})
    return {
        'formula': {
            'I_x': 'sum_{e,y} (1+d)^-(year_y - year_0) * sum_m omega_m * (c^S_{m,year_y} * '
                   'es_s_investment[e,y] + c^E_{m,year_y} * es_e_investment[e,y])',
            'source_expression': _loc('shared_energy_storage_data.py',
                                      'model.investment_cost = pe.Expression(expr=investment_cost)'),
            'objective_uses_it': _loc('shared_energy_storage_data.py',
                                      'model.objective = pe.Objective(sense=pe.minimize, expr=model.investment_cost + model.alpha)'),
            'salvage_in_I_x': False,
            'evaluation': 'production master built by shared_ess_data.build_master_problem(); '
                          'candidate loaded by shared_ess_data.load_candidate_solution_into_'
                          'master_model (' + _func_span(SED._load_candidate_solution_into_master_model)
                          + '); pe.value(model.investment_cost); no solve',
            'cross_check': 'p56a_oracle.investment_cost (' + _func_span(O.investment_cost) + ')',
        },
        'units': {'x_power': 'MVA', 'x_energy': 'MWh',
                  'unit_costs': 'EUR/MVA ("S, [€/MVA]") and EUR/MWh ("E, [€, MWh]") per '
                                'sheet header', 'I_x_and_budget': 'EUR (budget: case file '
                                '"budget" 1e6; SharedEnergyStorageParameters default comment '
                                '"1 M m.u.")',
                  'candidate_units_evidence': _loc('network_data.py',
                                                   "shared_energy_storages[shared_ess_idx].s = candidate_solution[node_id][year]['s'] / network_planning.network[year][day].baseMVA")
                                              + ' (candidate MVA divided by baseMVA)'},
        'case_file_values': {'budget': sed.params.budget, 'max_capacity': sed.params.max_capacity,
                             'min_energy_to_power_ratio': sed.params.min_energy_to_power_ratio,
                             'max_energy_to_power_ratio': sed.params.max_energy_to_power_ratio,
                             'discount_factor': sed.discount_factor,
                             'scenario_weights': list(sed.prob_market_scenarios)},
        'committed_cost_file': committed,
        'other_blobs_of_the_same_path_sensitivity': alternatives,
        'budget_and_max_capacity_in_the_model': {
            'benders_master_rows': {
                'max_capacity_row': _loc('shared_energy_storage_data.py',
                                         'model.energy_storage_maximum_capacity.add('),
                'max_capacity_row_variables': maxcap_vars,
                'max_capacity_bounds_energy_not_power': maxcap_vars == ['es_e_rated'],
                'ratio_rows': _loc('shared_energy_storage_data.py',
                                   'model.energy_storage_power_to_energy_factor.add(', 1),
                'budget_row': _loc('shared_energy_storage_data.py',
                                   'model.energy_storage_investment.add('),
                'budget_row_variables': budget_vars,
                'master_built_at': _loc('shared_resources_planning.py',
                                        'master_problem_model = planning_problem.shared_ess_data.build_master_problem()'),
                'master_builder': _func_span(SED._build_master_problem),
                'active_row_counts': {
                    'energy_storage_maximum_capacity': sum(1 for r in master.energy_storage_maximum_capacity.values() if r.active),
                    'energy_storage_power_to_energy_factor': sum(1 for r in master.energy_storage_power_to_energy_factor.values() if r.active),
                    'energy_storage_investment': sum(1 for r in master.energy_storage_investment.values() if r.active)},
            },
            'first_stage_check': _func_span(srp._check_candidate_first_stage_feasibility),
            'oracle_path_overrides': {
                'harness_budget_override': _loc('p515_g_g1_g4_admm_gates.py',
                                                'planning.shared_ess_data.params.budget = N.BUDGET'),
                'N_BUDGET_value': N.BUDGET,
                'N_BUDGET_defined_at': _loc('p514_n_instrumented_cstar.py', 'BUDGET, REL, CAP ='),
            },
            'production_attribute_scan': scan,
        },
    }


# ======================================================================================
#  Item 3 -- x = 0 at every node
# ======================================================================================
class _Interceptor:
    """Replaces `.optimize` on ONE planning instance's agents. Records the call and
    returns "no result" (None per block/node) -- production then treats the block as
    not solved and skips every post-solve read. Never calls a solver; never fabricates
    a solution."""

    def __init__(self):
        self.calls = []

    def network(self, holder, kind):
        def _stub(model, *args, **kwargs):
            self.calls.append(kind)
            return {year: {day: None for day in holder.days} for year in holder.years}
        return _stub

    def esso(self, sed):
        def _stub(models, *args, **kwargs):
            kind = 'esso_coordination' if kwargs.get('cycle') is not None else 'esso_init'
            self.calls.append(kind)
            return {node_id: None for node_id in sed.active_distribution_network_nodes}
        return _stub

    def counts(self):
        out = {k: 0 for k in DECLARED_INTERCEPTS}
        for c in self.calls:
            out[c] = out.get(c, 0) + 1
        return out


def _active_sess_rows(blk):
    out = {}
    for name in ('sess_converter_capability', 'sess_active_sum_limit', 'sess_soc_def',
                 'sess_soc_limit_upper', 'sess_soc_limit_lower', 'sess_soc_final'):
        comp = getattr(blk, name, None)
        out[name] = None if comp is None else {
            'rows': len(comp), 'active': sum(1 for r in comp.values() if r.active)}
    return out


def _step(steps, name, fn, where=None):
    try:
        result = fn()
        entry = {'step': name, 'ok': True}
        if where:
            entry['where'] = where
        if isinstance(result, dict):
            entry.update(result)
        steps.append(entry)
        return True
    except Exception as error:  # recorded, never swallowed silently
        steps.append({'step': name, 'ok': False, 'where': where,
                      'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()[-3000:]})
        return False


def _historical_zero_capacity_evidence():
    import glob
    scanned, zero_node_runs, all_zero_runs = 0, [], []
    for path in sorted(glob.glob(os.path.join(REPO, 'data', 'SRP1', 'Results', '**', 'g_*.json'),
                                 recursive=True)):
        scanned += 1
        try:
            g = json.load(open(path))
        except Exception:
            continue
        inst = g.get('instance') if isinstance(g, dict) else None
        if not isinstance(inst, dict):
            continue
        im = inst.get('investment_map')
        if im:
            zeros = [k for k, v in im.items() if float(v[0]) == 0.0 and float(v[1]) == 0.0]
            if zeros:
                rec = {'artifact': os.path.relpath(path, REPO), 'investment_map': im,
                       'zero_nodes': zeros, 'cycles_run': g.get('cycles_run'),
                       'converged_at_cycle': g.get('converged_at_cycle'),
                       'esso_diagnostics_grouping':
                           {k: v for k, v in (g.get('esso_complementarity_diagnostics_by_round') or {}).items()
                            if not isinstance(v, (list, dict))}}
                (all_zero_runs if len(zeros) == len(im) else zero_node_runs).append(rec)
        elif inst.get('s_mva') == 0 and inst.get('e_mwh') == 0:
            all_zero_runs.append({'artifact': os.path.relpath(path, REPO)})
    g3f = {}
    leak = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G3F_r2',
                        'leak_classification_g3_full_node7.jsonl')
    stdout = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G3F_r2', 'stdout_g3_full_node7.log')
    if os.path.exists(leak) and os.path.exists(stdout):
        per_node = {}
        for line in open(leak):
            r = json.loads(line)
            key = str(r['node_id'])
            per_node.setdefault(key, {'rounds': 0, 'mu_parsed': 0, 'n_active_cohort_periods': set()})
            per_node[key]['rounds'] += 1
            per_node[key]['mu_parsed'] += int(r.get('mu_unscaled') is not None)
            per_node[key]['n_active_cohort_periods'].add(r.get('n_active_cohort_periods'))
        text = open(stdout, errors='replace').read()
        g3f = {
            'artifacts': {os.path.relpath(leak, REPO): _sha256_file(leak),
                          os.path.relpath(stdout, REPO): _sha256_file(stdout)},
            'per_node_esso_rounds': {k: {'rounds': v['rounds'], 'mu_parsed': v['mu_parsed'],
                                         'n_active_cohort_periods': sorted(v['n_active_cohort_periods'])}
                                     for k, v in per_node.items()},
            'stdout_lines_shared_ess_did_not_converge':
                len(re.findall(r'Shared ESS .*did not converge', text)),
            'note': 'a leak record exists only for a round in which _optimize loaded a solution '
                    '(diagnostics are appended only on success)',
        }
    d1 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514D', 'd1_cell2.json')
    d1_rec = None
    if os.path.exists(d1):
        g = json.load(open(d1))
        d1_rec = {'artifact': os.path.relpath(d1, REPO), 'sha256': _sha256_file(d1),
                  'fields': {k: g.get(k) for k in ('cell', 'instance', 'cycles_run', 'recourse',
                                                  'converged_at_cycle', 'local_solve_failures')
                             if k in g}}
    return {'searched': 'every data/SRP1/Results/**/g_*.json (instance.investment_map / '
                        's_mva,e_mwh); P515G3F_r2 leak/stdout; P514D/d1_cell2.json',
            'g_json_files_scanned': scanned,
            'runs_with_some_zero_nodes': zero_node_runs,
            'runs_with_all_nodes_zero': all_zero_runs,
            'g3_full_zero_node_esso_outcomes': g3f,
            'track_d1_cell2_no_storage': d1_rec}


def item3_zero_investment(O, srp, SED, G, N, R, pe, eval_id, stdout_sink):
    steps = []
    state = {}
    arm_out = os.path.join(OUT_DIR, 'item3_arm_construct')
    construct_report = {}

    def construct():
        with redirect_stdout(stdout_sink):
            planning, sed, cand = G._construct_arm_planning(
                'p515s44_x0', arm_out, construct_report,
                investment_map={n: (0.0, 0.0) for n in (5, 7, 9)}, eval_id=eval_id,
                num_max_iters_override=500, apply_rho=False)
        state.update(planning=planning, sed=sed, cand=cand)
        return {'eval_id': eval_id, 'instance': construct_report.get('instance'),
                'is_zero_investment_candidate': srp._is_zero_investment_candidate(cand),
                'total_capacity': _jsonable(cand['total_capacity']),
                'shared_ess_initialization': planning.params.admm.shared_ess_initialization,
                'parallel_execution': planning.parallel_execution,
                'budget_in_force_on_oracle_copy': sed.params.budget,
                'fresh_planning_logs_dir_created': os.path.relpath(planning.logs_dir, REPO)}

    if not _step(steps, '1 oracle construction (_construct_arm_planning, D arm apply_rho=False, '
                        'investment_map all zero)', construct, _func_span(G._construct_arm_planning)):
        return {'steps': steps}
    planning, sed, cand = state['planning'], state['sed'], state['cand']
    params = planning.params.admm
    nodes = list(sed.active_distribution_network_nodes)

    _step(steps, '2 first-stage feasibility at x = 0',
          lambda: dict(zip(('feasible', 'reasons'),
                           srp._check_candidate_first_stage_feasibility(planning, cand))),
          _func_span(srp._check_candidate_first_stage_feasibility))

    interceptor = _Interceptor()
    tn = planning.transmission_network
    tn.optimize = interceptor.network(tn, 'tso')
    for dn in planning.distribution_networks.values():
        dn.optimize = interceptor.network(dn, 'dso')
    sed.optimize = interceptor.esso(sed)
    results = {}
    try:
        def admm_vars():
            cv, dv = srp.create_admm_variables(planning)
            state.update(cv=cv, dv=dv)
            return {'ess_consensus_nodes': sorted(cv['ess']['tso']['current'].keys())}
        ok = _step(steps, '3 create_admm_variables', admm_vars, _func_span(srp.create_admm_variables))

        def dso():
            with redirect_stdout(stdout_sink):
                dso_models, res = srp.create_distribution_networks_models(
                    planning.distribution_networks, state['cv'], cand['total_capacity'],
                    parallel_execution=planning.parallel_execution)
            state['dso'] = dso_models
            results['dso'] = res
            blk = dso_models[nodes[0]][INVEST_YEAR][list(planning.days)[0]]
            return {'dso_nodes_built': sorted(dso_models.keys()),
                    'sample_block_shared_es_s_rated_fixed': [pe.value(blk.shared_es_s_rated_fixed[e])
                                                             for e in blk.shared_energy_storages],
                    'sample_block_active_sess_rows': _active_sess_rows(blk),
                    'sample_block_expected_shared_ess_p_all_fixed_zero': all(
                        v.fixed and v.value == 0.0 for v in blk.expected_shared_ess_p.values())}
        ok = ok and _step(steps, '4 create_distribution_networks_models (sequential; optimize '
                                 'intercepted)', dso,
                          _func_span(srp.create_distribution_networks_models_sequential))

        def tso():
            with redirect_stdout(stdout_sink):
                tso_model, res = srp.create_transmission_network_model(
                    planning, state['cv'], cand['total_capacity'])
            state['tso'] = tso_model
            results['tso'] = res
            blk = tso_model[INVEST_YEAR][list(planning.days)[0]]
            return {'tso_blocks_built': sum(len(v) for v in tso_model.values()),
                    'sample_block_shared_es_s_rated_fixed': [pe.value(blk.shared_es_s_rated_fixed[e])
                                                             for e in blk.shared_energy_storages],
                    'sample_block_active_sess_rows': _active_sess_rows(blk),
                    'sample_block_expected_shared_ess_p_all_fixed_zero': all(
                        v.fixed and v.value == 0.0 for v in blk.expected_shared_ess_p.values())}
        ok = ok and _step(steps, '5 create_transmission_network_model (optimize intercepted)', tso,
                          _func_span(srp.create_transmission_network_model))

        def esso():
            with redirect_stdout(stdout_sink):
                esso_model, res = srp.create_shared_energy_storage_model(
                    sed, state['cv'], cand['investment'])
            state['esso'] = esso_model
            results['esso'] = res
            per_node = {}
            for n in nodes:
                m = esso_model[n]
                per_node[str(n)] = {
                    'cohort_inactive': dict(m._esso_cohort_inactive),
                    'pch_pdch_free': sum(1 for v in list(m.es_pch_per_unit.values())
                                         + list(m.es_pdch_per_unit.values()) if not v.fixed),
                    'active_rows': {name: sum(1 for r in getattr(m, name).values() if r.active)
                                    for name in ('rated_s_capacity_unit', 'rated_e_capacity_unit',
                                                 'energy_storage_limits',
                                                 'energy_storage_operation_agg',
                                                 'energy_storage_capacity_degradation',
                                                 'energy_storage_charging_discharging',
                                                 'energy_storage_cohort_pnet_share_h3')},
                    'es_s_investment_fixed': [pe.value(m.es_s_investment_fixed[y]) for y in m.years],
                }
            return {'per_node': per_node}
        ok = ok and _step(steps, '6 create_shared_energy_storage_model (build + candidate + '
                                 'TSO-request fixing; optimize intercepted)', esso,
                          _func_span(srp.create_shared_energy_storage_model))

        _step(steps, '7 production initialization-success check on the intercepted results '
                     '(expected False: nothing was solved -- trace artifact, not a finding)',
              lambda: {'_admm_local_solves_succeeded': srp._admm_local_solves_succeeded(
                  planning, results)},
              _func_span(srp._admm_local_solves_succeeded))

        def admm_prep():
            with redirect_stdout(stdout_sink):
                srp._prepare_distribution_objectives_for_admm(planning.distribution_networks,
                                                              state['dso'])
                srp._prepare_transmission_objectives_for_admm(tn, state['tso'])
                computed = srp._compute_common_admm_objective_scale(planning, state['tso'],
                                                                    state['dso'])
                scale, sig_c, sig_f = srp._resolve_common_admm_objective_scale(computed, params)
                al_scale = srp._resolve_esso_al_scale(planning, params, scale)[0]
                with R.patched_admm_objectives():
                    srp.update_distribution_models_to_admm(planning, state['dso'], params, scale)
                    srp.update_transmission_model_to_admm(planning, state['tso'], params, scale)
                srp.update_shared_energy_storage_model_to_admm(planning, state['esso'], params,
                                                               al_scale_esso=al_scale)
                srp._initialize_shared_ess_consensus(planning, state['cv'])
            state['al_scale'] = al_scale
            return {'objective_scale_computed_at_unsolved_point': computed,
                    'objective_scale_used': scale, 'al_scale_esso': al_scale,
                    'price_taker_branch_taken': params.shared_ess_initialization == 'price_taker'}
        if ok:
            ok = _step(steps, '8 ADMM objective preparation + ESSO AL + consensus initialization '
                              '(production order, _run_operational_planning init branch)',
                       admm_prep, _func_span(srp._run_operational_planning))

        def sref():
            ref = srp._admm_shared_ess_reference_mva(params)
            esso_norm = {}
            for year in sed.years:
                idx = {n: sed.get_shared_energy_storage_idx(n) for n in nodes}
                esso_norm[str(year)] = {str(n): srp._shared_ess_admm_normalization_mva(
                    sed.shared_energy_storages[year][idx[n]].s,
                    params.shared_ess_normalization_floor_mva, reference_mva=ref) for n in nodes}
            net = tn.network[INVEST_YEAR][list(planning.days)[0]]
            net_pu = [srp._shared_ess_admm_normalization_pu(
                s.s, net.baseMVA, params.shared_ess_normalization_floor_mva, reference_mva=ref)
                for s in net.shared_energy_storages]
            without_ref = srp._shared_ess_admm_normalization_mva(
                0.0, params.shared_ess_normalization_floor_mva, reference_mva=None)
            return {'reference_rating_mva_in_force': ref,
                    'floor_mva': params.shared_ess_normalization_floor_mva,
                    'esso_normalization_mva_by_year_node': esso_norm,
                    'tso_sample_normalization_pu': net_pu, 'tso_baseMVA': net.baseMVA,
                    'installed_s_seen_by_normalization': sorted({
                        sed.shared_energy_storages[y][i].s for y in sed.years
                        for i in range(len(sed.shared_energy_storages[y]))}),
                    'counterfactual_if_reference_None_mva': without_ref,
                    'reader': _func_span(srp._admm_shared_ess_reference_mva)}
        _step(steps, '9 S_ref normalization at x = 0 (the single source every ADMM '
                     'shared-ESS normalization site reads)', sref,
              _func_span(srp._shared_ess_admm_normalization_mva))

        def esso_structure():
            out = {}
            for n in nodes:
                m = state['esso'][n]
                circle_rows = [r for r in m.energy_storage_operation_agg.values()
                               if r.active and not r.equality]
                out[str(n)] = {
                    'circle_rows_active': len(circle_rows),
                    'es_pnet_fixed': sum(1 for v in m.es_pnet.values() if v.fixed),
                    'es_qnet_fixed': sum(1 for v in m.es_qnet.values() if v.fixed),
                    'rated_s_capacity_unit_rhs_params': [pe.value(m.es_s_investment_fixed[y])
                                                        for y in m.years],
                    'rated_s_per_unit_fixed_outside_lifetime': sum(
                        1 for v in m.es_s_rated_per_unit.values() if v.fixed),
                    'admm_objective_active': m.admm_objective.active,
                }
            return {'per_node': out, 'row_form': 'es_pnet[y,d,p]^2 + es_qnet[y,d,p]^2 <= '
                    'es_s_rated[y]^2 (' + _loc('shared_energy_storage_data.py',
                                               'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2')
                    + '); es_s_rated[y] == sum es_s_rated_per_unit == es_s_investment_fixed = 0 '
                    'through active equality rows'}
        if ok:
            _step(steps, '10 ESSO structure after ADMM preparation at x = 0', esso_structure)

        def capacities():
            with redirect_stdout(stdout_sink):
                caps = sed.get_updated_capacities(state['esso'])
            return {'sess_available_capacities': _jsonable(caps)}
        if ok:
            _step(steps, '11 get_updated_capacities (published to the networks each cycle)',
                  capacities)

        def coordination():
            with redirect_stdout(stdout_sink):
                srp.update_shared_energy_storages_coordination_model_and_solve(
                    planning, state['esso'], state['cv']['ess']['z'], state['dv']['ess']['esso'],
                    params, from_warm_start=False, cycle=1)
            m = state['esso'][nodes[0]]
            return {'p_req_sample': pe.value(m.p_req[0, 0, 0]),
                    'dual_p_req_sample': pe.value(m.dual_p_req[0, 0, 0])}
        if ok:
            _step(steps, '12 ESSO coordination update, cycle 1 (param setting; optimize '
                         'intercepted)', coordination,
                  _func_span(srp.update_shared_energy_storages_coordination_model_and_solve))

        def residuals():
            with redirect_stdout(stdout_sink):
                legacy = srp.get_admm_residual_metrics(planning, state['tso'], state['dso'],
                                                       state['esso'], state['cv'])
                boyd = srp.get_admm_boyd_residual_metrics(planning, state['tso'], state['dso'],
                                                          state['esso'], state['cv'],
                                                          state['dv'], params)
            ess_boyd = boyd.get('ess') if isinstance(boyd, dict) else None
            return {'legacy_metrics_keys': sorted(legacy.keys()) if isinstance(legacy, dict) else None,
                    'boyd_ess_channel_at_unsolved_state': _jsonable(
                        {k: v for k, v in (ess_boyd or {}).items()
                         if isinstance(v, (int, float, bool, type(None)))})}
        if ok:
            _step(steps, '13 ADMM residual metrics (legacy + Boyd) on the x = 0 state', residuals)

        def captures():
            out = {}
            out['efc_per_day_max'] = srp._get_admm_efc_per_day_max(state['esso'])
            cap = N.capture_esso(state['esso'], sed)
            out['p514n_capture_esso'] = {n: {'efc_per_day_max': v['efc_per_day_max'],
                                             'efc_per_day_min': v['efc_per_day_min'],
                                             'degradation_fraction_per_year_per_cohort_year':
                                                 v['degradation_fraction_per_year_per_cohort_year']}
                                         for n, v in cap.items()}
            out['complementarity_ratio_max'] = SED._get_complementarity_violation_ratio(
                sed, state['esso'])
            out['complementarity_per_node'] = {
                str(n): SED._complementarity_ratio_for_model(state['esso'][n])[:2] for n in nodes}
            with redirect_stdout(stdout_sink):
                out['available_capacity'] = _jsonable(sed.get_available_capacity(state['esso']))
                out['salvage_value'] = sed.get_salvage_value(state['esso'])
                out['salvage_value_results_keys'] = sorted(
                    sed.get_salvage_value_results(state['esso']).keys())
                out['feasibility_violation'] = sed.get_feasibility_violation(state['esso'])
                out['penalty_summary'] = _jsonable(srp._get_admm_penalty_summary(
                    state['tso'], state['dso'], state['esso']))
            tmp = tempfile.mkdtemp(prefix='p515s44_x0_capture_')
            try:
                hook_state = {'var_maps': {}, 'zL_checked': False}
                leak = G._capture_esso_solve(sed, state['esso'], {}, tmp, 'x0probe', hook_state)
                written = {f: sum(1 for _ in open(os.path.join(tmp, f)))
                           for f in sorted(os.listdir(tmp))}
            finally:
                shutil.rmtree(tmp)
            out['harness_capture_esso_solve'] = {
                'records_written_per_node_file': written,
                'leak_records': _jsonable([{k: r.get(k) for k in
                                            ('node_id', 'n_active_cohort_periods',
                                             'complementarity_ratio_max', 'parse_reason')}
                                           for r in leak]),
                'zL_precheck_marked_done': hook_state['zL_checked']}
            return out
        if ok:
            _step(steps, '14 evaluator captures on the x = 0 ESSO models (EFC, SoH/degradation, '
                         'complementarity detectors, available capacity, salvage, harness '
                         'per-solve capture)', captures)
    finally:
        del tn.optimize
        for dn in planning.distribution_networks.values():
            del dn.optimize
        del sed.optimize

    counts = interceptor.counts()
    intercept_check = {'declared': DECLARED_INTERCEPTS, 'observed': counts,
                       'exact_match': counts == DECLARED_INTERCEPTS}
    not_exercised = [
        'update_interface_power_flow_variables and every post-solve read of the initialization '
        '(require solved results)',
        'the ADMM main cycle (every DSO/TSO/ESSO solve), convergence certification, polish',
        'IPOPT behaviour on the ESSO converter-circle rows whose right-hand side is pinned to 0 '
        '(see step 10) and on the TSO/DSO blocks with the shared ESS fixed at 0 -- a solve '
        'question; see historical_evidence',
    ]
    campaign_harness = sorted(f for f in os.listdir(REPO)
                              if f.startswith('p515_s44_campaign_harness'))
    return {'steps': steps, 'interceptor_check': intercept_check,
            'not_exercised_zero_solve': not_exercised,
            'campaign_harness_files_present_at_run_time': campaign_harness,
            'historical_evidence': _historical_zero_capacity_evidence()}


# ======================================================================================
#  Item 4 -- ESSO capacity multipliers
# ======================================================================================
def item4_esso_multipliers(pe):
    import pickle
    sed_py = 'shared_energy_storage_data.py'
    rows = {
        'rated_s_capacity_unit': _loc(sed_py, 'model.rated_s_capacity_unit.add('),
        'rated_e_capacity_unit': _loc(sed_py, 'model.rated_e_capacity_unit.add('),
        'energy_storage_capacity_degradation_D_row_(E_inv_as_coefficient)':
            _loc(sed_py, "model.es_D_per_unit[y_inv, y] * (2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv])"),
        'downstream_rated_s_capacity_aggregate': _loc(sed_py, 'model.rated_s_capacity.add('),
        'downstream_energy_storage_limits_pch_pdch_le_s': _loc(sed_py, "'energy_storage_limits', y_inv, y, pch <= s_max"),
        'downstream_converter_circle': _loc(sed_py, 'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2'),
        'downstream_available_e_capacity_unit': _loc(sed_py, 'model.available_e_capacity_unit.add('),
    }
    suffix = {
        'dual_suffix_declared': _loc(sed_py, 'model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)', 2),
        'bound_multiplier_suffixes': _loc(sed_py, 'model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)', 2),
        'solution_load': _loc(sed_py, 'model.solutions.load_from(result)'),
        'network_dual_suffix_declared': _loc('network.py', 'model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)'),
        'harness_per_cycle_dual_families': _loc('p515_g_g1_g4_admm_gates.py', '_ESSO_DUAL_FAMILIES = ('),
        'network_benders_sensitivity_channel_retired': _loc('network_data.py', '_BENDERS_SENSITIVITY_CHANNEL_RETIRED = True'),
    }
    manifest = json.load(open(os.path.join(REPO, S42_DIR_REL, 'manifest_sha256.json')))
    mfiles = manifest.get('files', manifest)

    def _manifest_sha(rel):
        v = mfiles.get(rel)
        return v.get('sha256') if isinstance(v, dict) else v

    art = {}
    esso_rel = f'{S42_DIR_REL}/esso_models_s39_D.pkl'
    rc, tracked, _ = _git(['ls-files', '--', esso_rel])
    esso_models = pickle.load(open(os.path.join(REPO, esso_rel), 'rb'))
    per_node = {}
    for n, m in sorted(esso_models.items()):
        fam = {}
        for name in ('rated_s_capacity_unit', 'rated_e_capacity_unit', 'rated_s_capacity',
                     'rated_e_capacity', 'energy_storage_limits', 'energy_storage_operation_agg',
                     'energy_storage_capacity_degradation', 'available_e_capacity_unit'):
            cl = getattr(m, name)
            act = [c for c in cl.values() if c.active]
            vals = [m.dual[c] for c in act if c in m.dual]
            fam[name] = {'rows': len(cl), 'active': len(act), 'active_with_dual': len(vals),
                         'max_abs_dual': max(map(abs, vals)) if vals else None}
        cohort_rows = {}
        for idx, c in m.rated_s_capacity_unit.items():
            cohort_rows[f'rated_s_capacity_unit[{idx}]'] = m.dual.get(c)
        for idx, c in m.rated_e_capacity_unit.items():
            cohort_rows[f'rated_e_capacity_unit[{idx}]'] = m.dual.get(c)
        per_node[str(n)] = {
            'dual_suffix_entries': len(m.dual), 'zL_entries': len(m.ipopt_zL_out),
            'zU_entries': len(m.ipopt_zU_out),
            'investment_s_mva': [pe.value(m.es_s_investment_fixed[y]) for y in m.years],
            'investment_e_mwh': [pe.value(m.es_e_investment_fixed[y]) for y in m.years],
            'active_objective': [o.name for o in m.component_objects(pe.Objective, active=True)],
            'rho_esso': pe.value(m.rho) if hasattr(m, 'rho') else None,
            'admm_esso_al_scale': pe.value(m.admm_esso_al_scale) if hasattr(m, 'admm_esso_al_scale') else None,
            'families': fam, 'capacity_row_duals': cohort_rows,
            'row_index_map': 'rated_*_capacity_unit rows are added y_inv-major over the cohort '
                             'lifetime window: [1..3] = cohort 2025 (y=2025,2030,2035), '
                             '[4..5] = cohort 2030, [6] = cohort 2035 on this 3-year instance',
        }
    art['esso_models_s39_D.pkl'] = {
        'path': esso_rel, 'tracked_in_git': bool(tracked.strip()),
        'sha256': _sha256_file(os.path.join(REPO, esso_rel)),
        'manifest_sha256': _manifest_sha(esso_rel),
        'instance': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, 2025; oracle D; terminal '
                    '(cycle 139) ESSO models pickled after the run (run_admm_arm)',
        'per_node': per_node}

    cert_rel = f'{S42_DIR_REL}/certified_models.pkl'
    cert_sha = _sha256_file(os.path.join(REPO, cert_rel))
    certified = pickle.load(open(os.path.join(REPO, cert_rel), 'rb'))
    net_rows = ('sess_converter_capability', 'sess_active_sum_limit', 'sess_soc_limit_upper',
                'sess_soc_limit_lower', 'sess_soc_def', 'sess_soc_final')
    net_summary = {}
    for agent_key in ('tso', 'dso'):
        holders = ({'tso': certified['tso']} if agent_key == 'tso' else certified['dso'])
        for hname, blocks in holders.items():
            for year, days in blocks.items():
                for day, blk in days.items():
                    for name in net_rows:
                        cl = getattr(blk, name, None)
                        if cl is None:
                            continue
                        act = [c for c in cl.values() if c.active]
                        vals = [blk.dual[c] for c in act if c in blk.dual]
                        s = net_summary.setdefault(f'{agent_key}:{name}',
                                                   {'blocks': 0, 'active_rows': 0,
                                                    'active_rows_with_dual': 0, 'max_abs_dual': 0.0})
                        s['blocks'] += 1
                        s['active_rows'] += len(act)
                        s['active_rows_with_dual'] += len(vals)
                        if vals:
                            s['max_abs_dual'] = max(s['max_abs_dual'], max(map(abs, vals)))
    rc, tracked_c, _ = _git(['ls-files', '--', cert_rel])
    art['certified_models.pkl'] = {
        'path': cert_rel, 'tracked_in_git': bool(tracked_c.strip()), 'sha256': cert_sha,
        'manifest_sha256': _manifest_sha(cert_rel), 'matches_manifest': cert_sha == _manifest_sha(cert_rel),
        'content': 'certified TSO/DSO models at C*, D, cycle 139, pickled before polish '
                   '(p515_s42_exact_fix_rerun._persist_certified_models)',
        'network_capacity_row_duals': net_summary}

    cap_dir = os.path.join(REPO, S42_DIR_REL, 'esso_capture', 's39_D')
    cap = {}
    for n in (5, 7, 9):
        rel = f'{S42_DIR_REL}/esso_capture/s39_D/node{n}_cycle139.jsonl'
        comps, n_rec, none_duals, total_duals = {}, 0, 0, 0
        for line in open(os.path.join(REPO, rel)):
            r = json.loads(line)
            n_rec += 1
            for d in r.get('duals', []):
                total_duals += 1
                none_duals += int(d['dual'] is None)
                comps[d['component']] = comps.get(d['component'], 0) + 1
        cap[rel] = {'sha256': _sha256_file(os.path.join(REPO, rel)),
                    'manifest_sha256': _manifest_sha(rel), 'records': n_rec,
                    'dual_entries_by_component': comps, 'dual_entries_None': none_duals,
                    'dual_entries_total': total_duals}
    rc, tracked_cap, _ = _git(['ls-files', '--', f'{S42_DIR_REL}/esso_capture'])
    art['esso_capture_cycle139'] = {'tracked_in_git': bool(tracked_cap.strip()),
                                    'n_files_in_dir': len(os.listdir(cap_dir)), 'files': cap}
    return {'rows_linking_operation_to_installed_capacity': rows, 'suffix_and_load': suffix,
            'artifacts_read_no_solve': art}


# ======================================================================================
#  main
# ======================================================================================
def _write_manifest():
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    files = {}
    for root, _, names in os.walk(OUT_DIR):
        for name in sorted(names):
            p = os.path.join(root, name)
            if p == path:
                continue
            files[os.path.relpath(p, REPO)] = {'sha256': _sha256_file(p),
                                              'bytes': os.path.getsize(p)}
    rc, head, _ = _git(['rev-parse', 'HEAD'])
    manifest = {'stage': STAGE, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                'git_HEAD': head.strip(), 'files': files}
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f'wrote {path} ({len(files)} files)')


def main():
    os.chdir(REPO)
    for name in OUTPUT_FILES:
        if os.path.exists(os.path.join(OUT_DIR, name)):
            raise RuntimeError(f'refusing to overwrite existing artifact {name}')
    os.makedirs(OUT_DIR, exist_ok=True)
    preflight = _preflight_no_concurrent_harness()
    spec = json.load(open(SPEC_PATH))
    started = datetime.now(timezone.utc)
    eval_id = 'p515s44_confirm_x0_' + started.strftime('%Y%m%dT%H%M%SZ')

    guard = SolveProfileGuard(permitted=(), label='P5.15 S44 Addendum 26 confirmations').install()
    stdout_sink = io.StringIO()
    try:
        with redirect_stdout(stdout_sink):
            import pyomo.environ as pe
            import p56a_oracle as O
            import shared_resources_planning as srp
            import shared_energy_storage_data as SED
            import p514_n_instrumented_cstar as N
            import p58_rescale as R
            import p515_g_g1_g4_admm_gates as G
            baseline = O.load_baseline()
        planning = baseline['planning']

        print('[S44-A26] item 1 ...', flush=True)
        item1 = item1_cost_file_provenance(planning, SED)
        alt_blobs = {}
        for blob, info in item1['distinct_blobs_across_refs'].items():
            if not info['is_HEAD_blob']:
                _, data, _ = _git(['cat-file', 'blob', blob], binary=True)
                alt_blobs[blob] = data
        print('[S44-A26] item 2 ...', flush=True)
        item2 = item2_investment_cost(planning, O, srp, SED, G, N, pe, alt_blobs)
        print('[S44-A26] item 3 ...', flush=True)
        item3 = item3_zero_investment(O, srp, SED, G, N, R, pe, eval_id, stdout_sink)
        print('[S44-A26] item 4 ...', flush=True)
        item4 = item4_esso_multipliers(pe)
    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    guard_record = {'permitted': [], 'counts': dict(guard.counts), 'verify_0_failures': failures}
    if failures:
        raise RuntimeError(f'solve-profile guard: {failures}')
    if not item3['interceptor_check']['exact_match']:
        print('[S44-A26] WARNING: interceptor count differs from the declared count: '
              f"{item3['interceptor_check']}", flush=True)

    common = {'stage': STAGE, 'authority': AUTHORITY,
              'spec_sha256': _sha256_file(SPEC_PATH),
              'spec_key': spec['addendum26_confirmations_zero_solve'],
              'started_utc': started.isoformat(), 'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
              'script_sha256': _sha256_file(os.path.abspath(__file__)),
              'solve_profile_guard': guard_record, 'preflight': preflight}
    payloads = {
        'item1_cost_file_provenance.json': {**common, 'result': item1},
        'item2_investment_cost_and_budget_slack.json': {**common, 'result': item2},
        'item3_zero_investment_evaluability.json': {**common, 'result': item3},
        'item4_esso_capacity_multipliers.json': {**common, 'result': item4},
    }
    for name, payload in payloads.items():
        with open(os.path.join(OUT_DIR, name), 'w') as fh:
            json.dump(_jsonable(payload), fh, indent=1, default=str)
    with open(os.path.join(OUT_DIR, 'production_stdout_capture.log'), 'w') as fh:
        fh.write(stdout_sink.getvalue())

    print('[S44-A26] guard counts', guard.counts, 'verify(0) failures', failures)
    print('[S44-A26] item 1:', item1['provenance_verdict']['statement'])
    for name, r in item2['committed_cost_file'].items():
        print(f"[S44-A26] item 2 committed {name}: I={r['I_x_eur_master_expression']:.2f} "
              f"slack={r['budget_slack_B_minus_I_eur']:.2f} feasible={r['first_stage_feasible']} "
              f"({r['first_stage_reasons']})")
    for blob, alt in item2['other_blobs_of_the_same_path_sensitivity'].items():
        for name in ('paper_plan', 'c_star', 'lattice_c_star', 'lattice_paper_plan'):
            r = alt['candidates'][name]
            print(f"[S44-A26] item 2 blob {blob[:8]} {name}: I={r['I_x_eur_master_expression']:.2f} "
                  f"slack={r['budget_slack_B_minus_I_eur']:.2f}")
    print('[S44-A26] item 3 steps:', [(s['step'][:40], s['ok']) for s in item3['steps']])
    print('[S44-A26] item 3 interceptor:', item3['interceptor_check'])
    print('[S44-A26] done')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        _write_manifest()
        sys.exit(0)
    sys.exit(main())
