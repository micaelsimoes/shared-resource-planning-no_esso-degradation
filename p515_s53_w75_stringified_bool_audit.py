"""P5.15 W75 -- zero-solve audit of stringified booleans ("True"/"False" JSON strings) in artifacts.
READ-ONLY, ZERO SOLVES (SolveProfileGuard(permitted=()) armed at import, verify(0) at exit). Nothing is fixed,
modified, deleted or re-run; output goes to a NEW write-once directory.

Concern (W74 unexpected finding 2): `p515_g_g1_g4_admm_gates.py` serialises with `json.dump(..., default=str)`, so a
numpy bool reaching it is written as the STRING "True"/"False"; a reader that tests such a value by truthiness reads
"False" as True (inversion). This script records, as evidence:

  A  census: every *.json / *.jsonl under the repository root (excluding .git/ and .claude/worktrees/), tracked or
     not, parsed; every value that is exactly the string "True" or "False" is counted per (file, field, value).
     Field = the innermost dict key above the value (list elements inherit their parent key). Keys that are Pyomo
     index reprs (start with "(", a digit or "-") are collapsed to "<index-key>" and their distinct count recorded.
     Hit files are sha256-hashed at read time.
  B  type census of the two flags under audit (determinate_at_gt_error_bar, objective_convergence), plus
     residual_convergence and cycle_convergence as controls: per file, count of JSON bool True/False, string
     "True"/"False", null, other.
  C  consumers: every *.py in the working tree (tracked + untracked, excluding .claude/worktrees/) whose text contains
     each stringified field name as a quoted literal, and every ref (refs/heads, refs/remotes) that is NOT an
     ancestor of HEAD searched with `git grep` for the two audited flags. The declared consumer table CONSUMERS below
     (file, line, needle, mode, inputs) is verified line-by-line (the cited line must contain the needle), and for
     each consumer that reads a FILE, the string-valued count of the field in its declared inputs is computed from A/B.
  D  reports: every *.md (tracked + untracked, excluding .claude/worktrees/) containing either flag name, or citing
     a directory in which B found a stringified flag.

Formulas (preserved here):
  inverting_exposure(consumer) = number of values in the consumer's declared input files for which the consumer's
      reading mode returns the opposite of the intended boolean:
        truthiness (`if x`, `and x`, `sum(1 for .. if x)`):  "False" strings  (bool("False") is True)
        strict (`x is True`, `x == True`):                   "True" strings   ("True" is True -> False)
        strict (`x is False`, `x == False`):                 "False" strings
        pass-through copy / equality between two artifacts / hash:  0 (type preserved or compared like-for-like)
  a consumer is INVERTING when its mode can invert a stringified value, and EXPOSED when inverting_exposure > 0.

COMMAND (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w75_stringified_bool_audit.py \
      > data/SRP1/Results/P515S53/stringified_bool_audit_w75_launch.log 2>&1
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W75 stringified-bool audit (read-only)').install()

OUT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'stringified_bool_audit_w75')
OUT_JSON = os.path.join(OUT, 'stringified_bool_audit_w75.json')
OUT_MANIFEST = os.path.join(OUT, 'stringified_bool_audit_w75_manifest_sha256.json')
LAUNCH_LOG = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'stringified_bool_audit_w75_launch.log')
EXCLUDE_TOP = ('.git', os.path.join('.claude', 'worktrees'))
AUDITED = ('determinate_at_gt_error_bar', 'objective_convergence')
CONTROLS = ('residual_convergence', 'cycle_convergence')
INDEX_KEY = re.compile(r'^[\(\d\-]')

# Producers: (file, line, needle, fields, mechanism). Verified line-by-line.
PRODUCERS = [
    ('p515_g_g1_g4_admm_gates.py', 1431, 'default=str', ['objective_convergence'],
     'g_<label>.json (g_s39_D.json, g_s51limit_*.json) = json.dump(report, default=str); report.cycle_trajectory rows '
     'carry production history; objective_convergence is a numpy bool in premium-active (alpha > 0) runs'),
    ('shared_resources_planning.py', 3304, 'objective_convergence = (objective_change_abs <= objective_tolerance)',
     ['objective_convergence'], 'origin of the numpy bool (comparison of numpy floats); written to history at 3464'),
    ('p515_g_g1_g4_admm_gates.py', 3120, "'determinate_at_gt_error_bar': (", ['determinate_at_gt_error_bar'],
     '_s34_system_cost_vs_references: s31c/s32/s33e2 legs; numpy bool when the run\'s gross cost is numpy'),
    ('p515_g_g1_g4_admm_gates.py', 3680, "'determinate_at_gt_error_bar': (", ['determinate_at_gt_error_bar'],
     '_s35ref_system_cost_vs_s34: s34 leg'),
    ('p515_g_g1_g4_admm_gates.py', 3864, 'default=str', ['determinate_at_gt_error_bar'],
     'write_boyd_terminal_s35ref: boyd_terminal.json = json.dump(payload, default=str)'),
    ('p515_s44_campaign_harness.py', 4612, '_write_once_json(os.path.join(eval_dir, POST_CERTIFICATION_FILE), pc)',
     ['pass'], 'post_certification.json; pre-W74 hook default=str (W74 replaced it with _json_default at 382)'),
    ('p515_s31c_zero_solve_checks.py', 472, 'default=str',
     ['pass', 'cancels_to_rounding', 'reconciles', 'sum_matches_residual_formula'], 'numpy comparisons, default=str'),
    ('p515_s36_a17_parta_dual_reconstruction.py', 489, 'default=str',
     ['passes_1e-9_relative', 'overall_pass_1e-9_relative'], 'numpy comparisons (e.g. 401), default=str'),
    ('p515_s47_tso_marginal_cost.py', 1195, 'default=_default', ['from_warm_start'],
     '_default = str() for non-tuples; value sits among string-typed snapshot metadata (year/day/cycle strings)'),
    ('p515_s49_flex_pq_split.py', 1437, 'default=_default', ['from_warm_start'], 'same hook as tso_marginal_cost'),
    ('p515_s53_w74_polish_feasibility.py', 162, "pce['gate_pass_raw'] = repr(", ['gate_pass_raw'],
     'deliberate repr() of the raw committed value'),
    ('p515_s53_alpha_row_recompute.py', 474, "'full_gate_pass': g['pass']", ['full_gate_pass'],
     'pass-through of the post_certification.json string, type recorded alongside (full_gate_pass_type)'),
    ('p515_s53_w74_harness_fix_checks.py', 349, "pre_w74['fail_case'] == 'False'", ['pass', 'fail_case'],
     'deliberate negative-control fixture of the pre-W74 hook'),
    ('p515_s47_case_file_baseline_check.py', 158, "comps['Blocks:active']", ["'ESSO, Operational Planning'"],
     'deliberate repr() state serialisation (not a numpy defect)'),
    ('p513_c_param_move_gate.py', 100, 'repr(v.fixed)', ['<index-key>'],
     'deliberate repr() state serialisation (untracked P513C/P513D state_*.json; P513D written by p513_e_gated_capture'
     '.py:95 with the same model_state)'),
    ('p45_seed2026_smoke_test.py', 103, "'warm_start': warm", ['warm_start'],
     'regex group parsed from log text (not a numpy defect); reused by p5_reduced_planning_baseline.py:236'),
]

# Consumers: (file, line, needle, field, mode, inputs(glob list, repo-relative) or None for in-memory).
CONSUMERS = [
    # objective_convergence
    ('shared_resources_planning.py', 3332, 'elif objective_convergence:', 'objective_convergence', 'truthiness',
     None, 'in-memory value (numpy/Python bool), prints a message only'),
    ('shared_resources_planning.py', 3345, 'cycle_convergence = boyd_all_pass and local_solves_ok',
     'objective_convergence', 'not-read', None, 'the stop rule does not read objective_convergence (diagnostic-only)'),
    ('shared_resources_planning.py', 9643, "('objective_convergence', 'Recourse Converged', 'General')",
     'objective_convergence', 'excel-cell', None, 'openpyxl writes a numpy bool as a numeric 0/1 cell, not a string'),
    ('p515_s32_supplementary.py', 43, "r.get('objective_convergence')", 'objective_convergence', 'truthiness',
     ['data/SRP1/Results/P515S32_run/g_baseline.json'], 'legacy_stop_cycle'),
    ('p515_s32_supplementary.py', 54, "r.get('objective_convergence')", 'objective_convergence', 'truthiness',
     ['data/SRP1/Results/P515S32_run/g_baseline.json'], 'first_objective_convergence_cycle'),
    ('p59_c_criteria.py', 190, "c.get('objective_convergence')", 'objective_convergence', 'truthiness',
     ['data/SRP1/Results/P59/' + n for n in ('p59_a_sweep.json', 'p59_a_refine.json', 'p59_b_adaptive.json',
                                              'p59_d_replay.json', 'p59_d2_signal_base.json',
                                              'p59_d3_largesignal_base.json', 'p59_e2_flexibility.json')],
     'n_changes_accepted_as_converged'),
    ('p510_e_criteria.py', 193, "c.get('objective_convergence')", 'objective_convergence', 'truthiness',
     ['data/SRP1/Results/P510/' + n for n in ('p510_a_state.json', 'p510_b_fixedrho_pf300.json',
                                               'p510_b_fixedrho_pf500.json', 'p510_b_fixedrho_pf1000.json',
                                               'p510_c_endpoint_pf300.json', 'p510_c_endpoint_pf1000.json',
                                               'p510_f_replay_base.json',
                                               'p510_f_replay_se_node5_2025_-10pct.json',
                                               'p510_f_replay_se_node9_2025_-10pct.json')],
     'n_changes_accepted_as_converged'),
    ('p59_c_criteria.py', 141, "cycle.get('objective_convergence')", 'objective_convergence', 'pass-through', None, ''),
    ('p510_e_criteria.py', 144, "cycle.get('objective_convergence')", 'objective_convergence', 'pass-through', None, ''),
    ('p58_eval.py', 178, "last.get('objective_convergence')", 'objective_convergence', 'pass-through', None, ''),
    ('p59_rho.py', 231, "last.get('objective_convergence')", 'objective_convergence', 'pass-through', None, ''),
    ('p512_a_cold_rescaled_convergence.py', 90, "entry.get('objective_convergence')", 'objective_convergence',
     'pass-through', None, ''),
    ('p510_a_state.py', 151, "first.get('objective_convergence')", 'objective_convergence', 'pass-through', None,
     'and an f-string print at 161'),
    ('p515_s32_zero_solve_checks.py', 704, "'objective_convergence' not in gating_line", 'objective_convergence',
     'source-text', None, 'checks production SOURCE, not a value'),
    ('p515_s33e2_zero_solve_checks.py', 538, "'objective_convergence' not in gating_line", 'objective_convergence',
     'source-text', None, 'checks production SOURCE, not a value'),
    ('p515_s52_pilot_campaign.py', 1340, 'type(a_rows[i].get(field)) is type(b_rows[i].get(field))', '*',
     'equality-between-artifacts', None, 'W47 two-cycle compare: RECORD_TRAJECTORY_FIELDS (no objective_convergence)'
     ' type-and-value equality; _row_diffs every field, equality'),
    # post_certification.json pass
    ('p515_s53_alpha_row_recompute.py', 474, "'full_gate_pass': g['pass']", 'pass', 'pass-through', None,
     'copies the string and records its type (str)'),
    ('p515_s53_w74_harness_fix_checks.py', 346, "gate['pass'] == 'True'", 'pass', 'string-equality', None,
     'deliberately asserts the committed value IS the string'),
    # others
    ('p515_s31c_zero_solve_checks.py', 456, "return bool(v['pass'])", 'pass', 'truthiness', None,
     'in-memory numpy bool before serialisation'),
    ('p515_s36_a17_parta_dual_reconstruction.py', 436, "norm_cross_val[agent]['passes_1e-9_relative']",
     'passes_1e-9_relative', 'truthiness', None, 'in-memory before serialisation'),
    ('p515_s47_case_file_x0_diff_analysis.py', 70, 'return ast.literal_eval(text)', "'ESSO, Operational Planning'",
     'literal_eval', None, 'repr strings parsed back to real bools; Blocks:active compared by equality only'),
    ('p513_c_param_move_gate.py', 198, 'if base[key] != cand[key]:', '<index-key>', 'equality-between-artifacts',
     None, 'hash / equality comparison of state files'),
]

REPORT_DIR_MARKERS = ('P515S51/2x2_limit_gate', 'campaign_s52_pilot_nopersist', 'campaign_s52_pilot_repro_nopersist',
                      'campaign_s53_alpha_row_smoke', 'campaign_s53_alpha_row_v25', 'g_s51limit', 'boyd_terminal')


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    h = hashlib.sha256()
    with open(_abs(rel), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def _walk_files(suffixes):
    out = []
    for dp, dn, fn in os.walk(REPO):
        rel = os.path.relpath(dp, REPO)
        if any(rel == e or rel.startswith(e + os.sep) for e in EXCLUDE_TOP):
            dn[:] = []
            continue
        out += [os.path.relpath(os.path.join(dp, f), REPO) for f in fn if f.endswith(suffixes)]
    return sorted(out)


def _load_docs(rel):
    with open(_abs(rel), 'rb') as handle:
        raw = handle.read()
    text = raw.decode('utf-8', errors='replace')
    try:
        return [json.loads(text)], raw, 'json'
    except Exception:  # noqa: BLE001 -- JSONL, or a .json holding JSON lines
        docs, bad = [], 0
        for line in text.splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    bad += 1
        return docs, raw, f'jsonl(bad_lines={bad})'


def _walk(obj, key, strings, types):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, k, strings, types)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, key, strings, types)
    else:
        if isinstance(obj, str) and obj in ('True', 'False'):
            strings[(key, obj)] += 1
        if key in AUDITED + CONTROLS:
            if isinstance(obj, bool):
                t = f'bool:{obj}'
            elif isinstance(obj, str) and obj in ('True', 'False'):
                t = f'str:{obj}'
            elif obj is None:
                t = 'null'
            else:
                t = f'other:{type(obj).__name__}'
            types[(key, t)] += 1


def census():
    files = _walk_files(('.json', '.jsonl'))
    tracked = set(_git('ls-files', '-z').split('\0'))
    hits, flag_types, parse = [], [], Counter()
    for rel in files:
        with open(_abs(rel), 'rb') as handle:
            raw = handle.read()
        if b'"True"' not in raw and b'"False"' not in raw and not any(k.encode() in raw for k in AUDITED + CONTROLS):
            continue
        docs, _raw, mode = _load_docs(rel)
        parse[mode.split('(')[0]] += 1
        strings, types = Counter(), Counter()
        for d in docs:
            _walk(d, None, strings, types)
        if strings:
            per_field, index_keys = Counter(), set()
            for (k, v), n in strings.items():
                kk = k
                if isinstance(k, str) and INDEX_KEY.match(k):
                    index_keys.add(k)
                    kk = '<index-key>'
                per_field[f'{kk}|{v}'] += n
            hits.append({'path': rel, 'tracked': rel in tracked, 'sha256_at_read': hashlib.sha256(raw).hexdigest(),
                         'parse': mode, 'string_counts': dict(sorted(per_field.items())),
                         'n_distinct_index_keys': len(index_keys)})
        if types:
            flag_types.append({'path': rel, 'tracked': rel in tracked,
                               'types': {f'{k}|{t}': n for (k, t), n in sorted(types.items())}})
    return files, hits, flag_types, dict(parse)


def totals(hits, flag_types):
    by_field = {}
    for h in hits:
        for kv, n in h['string_counts'].items():
            field, val = kv.rsplit('|', 1)
            e = by_field.setdefault(field, {'True': 0, 'False': 0, 'files': 0, 'files_tracked': 0,
                                            'False_tracked': 0, 'True_tracked': 0})
            e[val] += n
            if h['tracked']:
                e[f'{val}_tracked'] += n
        for field in {kv.rsplit('|', 1)[0] for kv in h['string_counts']}:
            by_field[field]['files'] += 1
            by_field[field]['files_tracked'] += int(h['tracked'])
    type_totals = Counter()
    for f in flag_types:
        for kt, n in f['types'].items():
            type_totals[f"{kt}|{'tracked' if f['tracked'] else 'untracked'}"] += n
    return by_field, dict(sorted(type_totals.items()))


def verify_lines(table):
    out = []
    for row in table:
        rel, line, needle = row[0], row[1], row[2]
        try:
            with open(_abs(rel)) as handle:
                text = handle.read().splitlines()[line - 1]
            ok = needle in text
        except Exception as error:  # noqa: BLE001
            text, ok = f'{type(error).__name__}: {error}', False
        out.append({'file': rel, 'line': line, 'needle': needle, 'line_text': text.strip(), 'verified': ok})
    return out


def consumer_exposure(flag_types, hits):
    types_by_path = {f['path']: f['types'] for f in flag_types}
    strings_by_path = {h['path']: h['string_counts'] for h in hits}
    out = []
    for (rel, line, needle, field, mode, inputs, note) in CONSUMERS:
        e = {'file': rel, 'line': line, 'field': field, 'mode': mode, 'note': note, 'inputs': inputs}
        inverting_mode = mode in ('truthiness',)
        e['inverting_mode'] = inverting_mode
        if inputs:
            per_input = {}
            for p in inputs:
                present = os.path.isfile(_abs(p))
                t = types_by_path.get(p, {})
                s = strings_by_path.get(p, {})
                per_input[p] = {'present': present,
                                'str_False': s.get(f'{field}|False', 0), 'str_True': s.get(f'{field}|True', 0),
                                'types': {k: v for k, v in t.items() if k.startswith(field + '|')}}
            e['per_input'] = per_input
            e['inverting_exposure'] = sum(v['str_False'] for v in per_input.values()) if inverting_mode else 0
        else:
            e['inverting_exposure'] = 0
            e['per_input'] = 'in-memory or not a value reader'
        e['classification'] = ('INVERTING' if inverting_mode and inputs else 'SAFE')
        e['exposed'] = e['inverting_exposure'] > 0
        out.append(e)
    return out


def static_search(hit_fields):
    py = [p for p in _walk_files(('.py',))]
    names = sorted(set(hit_fields) | set(AUDITED))
    refs = {}
    for name in names:
        if name == '<index-key>':
            continue
        lits = (f"'{name}'", f'"{name}"') if not name.startswith("'") else (name,)
        found = []
        for rel in py:
            try:
                with open(_abs(rel), errors='replace') as handle:
                    txt = handle.read()
            except Exception:  # noqa: BLE001
                continue
            if any(lit in txt for lit in lits):
                lines = [i + 1 for i, ln in enumerate(txt.splitlines()) if any(lit in ln for lit in lits)]
                found.append({'file': rel, 'lines': lines})
        refs[name] = found
    head = _git('rev-parse', 'HEAD').strip()
    all_refs = [r for r in _git('for-each-ref', '--format=%(refname)', 'refs/heads', 'refs/remotes').split() if r]
    non_ancestor = [r for r in all_refs
                    if subprocess.run(['git', 'merge-base', '--is-ancestor', r, head], cwd=REPO).returncode != 0]
    branch_hits = {}
    for r in non_ancestor:
        res = _git('grep', '-l', '-e', AUDITED[0], '-e', AUDITED[1], r, '--', '*.py').split()
        if res:
            branch_hits[r] = res
    return {'py_files_searched': len(py), 'literal_references': refs, 'refs_total': len(all_refs),
            'non_ancestor_refs_searched': non_ancestor, 'non_ancestor_refs_with_audited_flag_in_py': branch_hits}


def report_search(stringified_dirs):
    md = _walk_files(('.md',))
    tracked = set(_git('ls-files', '-z').split('\0'))
    out = {'md_files_searched': len(md), 'mention_audited_flag': [], 'cite_stringified_artifact_dir': []}
    for rel in md:
        with open(_abs(rel), errors='replace') as handle:
            txt = handle.read()
        for flag in AUDITED:
            lines = [i + 1 for i, ln in enumerate(txt.splitlines()) if flag in ln]
            if lines:
                out['mention_audited_flag'].append({'file': rel, 'tracked': rel in tracked, 'flag': flag,
                                                    'lines': lines})
        for marker in stringified_dirs:
            lines = [i + 1 for i, ln in enumerate(txt.splitlines()) if marker in ln]
            if lines:
                out['cite_stringified_artifact_dir'].append({'file': rel, 'tracked': rel in tracked,
                                                             'marker': marker, 'lines': lines})
    return out


def main():
    started = time.time()
    if os.path.exists(_abs(OUT)):
        raise SystemExit(f'REFUSED: {OUT} exists (write-once)')
    files, hits, flag_types, parse = census()
    by_field, type_totals = totals(hits, flag_types)
    audited_str_files = sorted({f['path'] for f in flag_types
                                if any('|str:' in k and k.split('|')[0] in AUDITED for k in f['types'])})
    payload = {
        'task': 'P5.15 W75: zero-solve audit of stringified booleans (report only, nothing fixed)',
        'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD').strip(),
        'script_sha256': _sha(os.path.basename(__file__)),
        'scope': {'root': 'repository working tree', 'excluded': list(EXCLUDE_TOP),
                  'json_files_scanned': len(files), 'parse_modes': parse},
        'A_string_census_per_file': hits,
        'A_string_totals_per_field': by_field,
        'B_flag_type_census_per_file': flag_types,
        'B_flag_type_totals': type_totals,
        'B_files_with_stringified_audited_flag': audited_str_files,
        'C_producers_verified': [dict(v, fields=p[3], mechanism=p[4]) for v, p in zip(verify_lines(PRODUCERS),
                                                                                      PRODUCERS)],
        'C_consumer_lines_verified': verify_lines(CONSUMERS),
        'C_consumers': consumer_exposure(flag_types, hits),
        'C_static_search': static_search({kv.rsplit('|', 1)[0] for h in hits for kv in h['string_counts']}),
        'D_report_search': report_search(REPORT_DIR_MARKERS),
        'formulas': {'inverting_exposure': __doc__.split('Formulas (preserved here):')[1].split('COMMAND')[0]},
    }
    failures = GUARD.verify(0)
    payload['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': failures}
    payload['wall_s'] = time.time() - started
    os.makedirs(_abs(OUT))
    with open(_abs(OUT_JSON), 'x') as handle:
        json.dump(payload, handle, indent=1)
    m = {OUT_JSON: _sha(OUT_JSON), os.path.basename(__file__): _sha(os.path.basename(__file__))}
    for h in hits:
        m[h['path']] = h['sha256_at_read']
    with open(_abs(OUT_MANIFEST), 'x') as handle:
        json.dump(m, handle, indent=1)
    for field, e in sorted(by_field.items()):
        print(f'[W75] {field}: str True {e["True"]} / str False {e["False"]} in {e["files"]} files '
              f'({e["files_tracked"]} tracked; False tracked {e["False_tracked"]})', flush=True)
    for k, n in type_totals.items():
        print(f'[W75] type {k}: {n}', flush=True)
    bad = [v for v in payload['C_producers_verified'] + payload['C_consumer_lines_verified'] if not v['verified']]
    print(f'[W75] unverified citations: {bad}', flush=True)
    for c in payload['C_consumers']:
        print(f"[W75] consumer {c['file']}:{c['line']} {c['field']} {c['mode']} -> {c['classification']} "
              f"exposure {c['inverting_exposure']}", flush=True)
    print(f"[W75] non-ancestor refs with audited flag: {payload['C_static_search']['non_ancestor_refs_with_audited_flag_in_py']}",
          flush=True)
    print(f'[W75] wrote {OUT_JSON}, {OUT_MANIFEST} ({len(m)} entries); guard {dict(GUARD.counts)} verify(0) {failures};'
          f' wall {payload["wall_s"]:.1f}s', flush=True)
    GUARD.uninstall()
    sys.exit(0 if not failures and not bad else 1)


if __name__ == '__main__':
    main()
