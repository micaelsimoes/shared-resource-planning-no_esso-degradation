"""P5.15 W76 item 2 -- ZERO-SOLVE, confined-change checks of the `_json_default` hook in `p515_g_g1_g4_admm_gates.py`
(W74's pattern, `p515_s44_campaign_harness._json_default`): every `json.dump` / `json.dumps` site of the module that
passed `default=str` now passes `default=_json_default`, so a numpy boolean is written as a JSON boolean instead of the
string "True"/"False". NOT a gate re-run (the SRP1 bitwise gate is not run: its arm-named eval directory under P56A/evals/
cannot be redirected, W61). Nothing committed is modified; every write goes under the write-once root OUT.

Checks:
  V1 AST equality. The pre-change module is the git blob at PRE_COMMIT (sha256-pinned). Normalisation, identical on both
     sides: the value of every `default=` keyword of a `json.dump` / `json.dumps` call is replaced by one placeholder
     name; on the new side the single added top-level `def _json_default` (the declared addition) is removed. The two
     normalised `ast.dump`s must be IDENTICAL; any other difference is a FAIL. Also: pre has exactly N_SITES json
     `default=` keywords, all `str`; new has exactly N_SITES, all `_json_default`; the per-site enclosing function and
     call form (dump/dumps, other keywords) are unchanged; every non-json `default=` (max/min) is untouched; and a line
     diff shows only `default=str` -> `default=_json_default` substitutions plus the added helper block.
  V2 positive control. For EACH of the N_SITES sites the site's own `default=` expression (taken from the new AST) is
     evaluated in the imported module's namespace and must be `G._json_default`; the site's own call form and keywords
     (indent / separators) then serialise a payload carrying numpy booleans produced by numpy comparisons; the text is
     written under OUT (json.dump sites) or kept as a line (json.dumps sites) and read back: strict `is True` /
     `is False` must hold. The module-level writer `G._atomic_write_json` is additionally called directly (the only
     site whose writer is a standalone function; the other sites are inline in hooks that need a solve).
  V3 negative control. The pre-change hook (`str`, the site expression taken from the pre AST) serialises the same
     payload to the strings "True"/"False" at every site; strict reads fail on it.
  V4 non-bool values unchanged. (a) A type battery -- pathlib.Path, datetime/date, numpy float16/32, numpy int8/32/64,
     numpy uint8, numpy ndarray, set, frozenset, Decimal, complex, bytes, a Pyomo scalar Var, a custom object -- gives
     `_json_default(x) == str(x)` and byte-identical `json.dumps` text under both hooks at every site's keywords; JSON-
     native values (bool, None, int, float, numpy float64 as a float subclass, str, list, dict) never reach the hook and
     are byte-identical. (b) Evidence of the types actually present: every committed (git-tracked) artifact written by
     this module's writers is parsed and each string leaf is classified by the shape `str()` gives it (bool-like,
     int-like, float-like, repr-like "<...>", set-like "{...}", array-like "[...]", date-like); counts per file kind and
     key are recorded. A shape match is a CANDIDATE hook output, not proof (a genuine string can have the same shape).
  V5 pin blast radius (enumeration, zero loosening). (a) every historical blob sha256 of the module (all commits touching
     it) searched as a literal in every *.py of the working tree (excluding .git/ and .claude/worktrees/); (b) every *.py
     line naming the module file, with its enclosing assignment / call; (c) every string literal in any *.py that
     imports the module whose presence in the module source (whole module and each changed function) flips pre -> new
     -- i.e. every `'<literal>' in inspect.getsource(G...)` check whose result the edit can change; (d) every
     (file, line, needle) citation of the module in any *.py, checked pre and new; (e) committed JSON artifacts recording
     the pre-change sha256; (f) the pre-change blob resolves from git at PRE_COMMIT to the pinned sha256 (how committed
     evidence must be re-verified).

A `SolveProfileGuard(permitted=())` is installed BEFORE the module import and verified at exactly 0 at the end.

COMMAND (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w76_gates_json_default_checks.py \
      > data/SRP1/Results/P515S53/gates_json_default_w76_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w76_gates_json_default_checks.py --manifest
"""

import ast
import datetime as _dt
import decimal
import difflib
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W76 gates _json_default checks').install()

import numpy as np  # noqa: E402
import pyomo.environ as pe  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402

MODULE_REL = 'p515_g_g1_g4_admm_gates.py'
PRE_COMMIT = '9f00377a'   # HEAD before the W76 item 2 edit (module unchanged since 298e58f0)
PRE_SHA256 = '172260450ba167c13c8776881a01c1b0ee096baa79cb98fa2650f83385ad458b'
N_SITES = 21
ADDED_DEF = '_json_default'
PLACEHOLDER = '__W76_DEFAULT_HOOK__'

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'gates_json_default_w76')
OUT_JSON = os.path.join(OUT, 'gates_json_default_w76_checks.json')
OUT_MANIFEST = os.path.join(OUT, 'gates_json_default_w76_manifest_sha256.json')
LAUNCH_LOG_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'gates_json_default_w76_launch.log')

# File kinds written by this module's `default=` writers (by basename pattern), for V4(b).
G_WRITTEN_PATTERNS = (
    r'g_[^/]*\.json', r'boyd_terminal\.json', r'component_levels_terminal\.json',
    r'interface_settlement_detail_s31c\.json', r'interface_voltage_terminal\.json',
    r'recourse_jump_sidecar_[^/]*\.jsonl', r'ess_entry_stride_[^/]*\.jsonl', r'soh_floor_sidecar_[^/]*\.jsonl',
    r'pf_entry_stride_[^/]*\.jsonl', r'ess_exempt_until_state_[^/]*\.jsonl', r'leak_classification_[^/]*\.jsonl',
    r'network_failures_[^/]*\.jsonl', r'frozen_snapshots_[^/]*\.jsonl', r'esso_recovery_events_[^/]*\.jsonl',
    r'heartbeat_[^/]*\.json', r'terminal_storage_duals_[^/]*\.json', r'bitwise_identity_check_vs_run1\.json',
    r'[^/]*cycle0_lmp[^/]*\.json', r'esso_capture/node[^/]*\.jsonl')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(path):
    with open(path if os.path.isabs(path) else os.path.join(REPO, path), 'rb') as handle:
        return _sha_bytes(handle.read())


def _git(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def pre_source():
    blob = _git('show', f'{PRE_COMMIT}:{MODULE_REL}', binary=True)
    got = _sha_bytes(blob)
    if got != PRE_SHA256:
        raise RuntimeError(f'pre-change blob sha256 {got} != pinned {PRE_SHA256}')
    return blob.decode()


# ----------------------------------------------------------------------------------------------------------------------
def _is_json_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('dump', 'dumps')
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'json')


def _parents(tree):
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    return parents


def _enclosing(node, parents):
    chain = []
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chain.append(node.name)
    return '.'.join(reversed(chain)) or '<module>'


def json_default_sites(src):
    tree = ast.parse(src)
    parents = _parents(tree)
    sites = []
    for node in ast.walk(tree):
        if _is_json_call(node):
            for kw in node.keywords:
                if kw.arg == 'default':
                    sites.append({
                        'line': node.lineno, 'form': node.func.attr, 'enclosing': _enclosing(node, parents),
                        'hook': ast.get_source_segment(src, kw.value),
                        'other_keywords': {k.arg: ast.get_source_segment(src, k.value) for k in node.keywords
                                           if k.arg != 'default'},
                        'hook_node': kw.value})
    sites.sort(key=lambda s: s['line'])
    return tree, sites


def non_json_defaults(src):
    tree = ast.parse(src)
    return sorted((ast.unparse(n.func), ast.get_source_segment(src, kw.value), ast.unparse(n))
                  for n in ast.walk(tree) if isinstance(n, ast.Call) and not _is_json_call(n)
                  for kw in n.keywords if kw.arg == 'default')


class _Normalise(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if _is_json_call(node):
            for kw in node.keywords:
                if kw.arg == 'default':
                    kw.value = ast.Name(id=PLACEHOLDER, ctx=ast.Load())
        return node


def v1_ast(old_src, new_src):
    old_tree, old_sites = json_default_sites(old_src)
    new_tree, new_sites = json_default_sites(new_src)
    added = [n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == ADDED_DEF]
    pre_had = [n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == ADDED_DEF]
    new_tree.body = [n for n in new_tree.body if not (isinstance(n, ast.FunctionDef) and n.name == ADDED_DEF)]
    old_norm = ast.dump(_Normalise().visit(old_tree), include_attributes=False)
    new_norm = ast.dump(_Normalise().visit(new_tree), include_attributes=False)
    first_diff = None
    if old_norm != new_norm:
        i = next((k for k in range(min(len(old_norm), len(new_norm))) if old_norm[k] != new_norm[k]),
                 min(len(old_norm), len(new_norm)))
        first_diff = {'offset': i, 'pre': old_norm[max(0, i - 200):i + 200], 'new': new_norm[max(0, i - 200):i + 200]}
    # line diff: only default=str -> default=_json_default substitutions plus the added helper block
    old_lines, new_lines = old_src.splitlines(), new_src.splitlines()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    substitutions, inserted, other = 0, [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'replace' and (i2 - i1) == (j2 - j1) and all(
                old_lines[i1 + k].replace('default=str', 'default=_json_default') == new_lines[j1 + k]
                and old_lines[i1 + k] != new_lines[j1 + k] for k in range(i2 - i1)):
            substitutions += i2 - i1
        elif tag == 'insert':
            inserted.append({'new_lines': [j1 + 1, j2], 'text': new_lines[j1:j2]})
        else:
            other.append({'tag': tag, 'pre_lines': [i1 + 1, i2], 'new_lines': [j1 + 1, j2],
                          'pre': old_lines[i1:i2], 'new': new_lines[j1:j2]})
    helper_lines = ast.get_source_segment(new_src, added[0]).splitlines() if added else None
    inserted_is_helper = (len(inserted) == 1 and helper_lines is not None
                          and [x for x in inserted[0]['text'] if x.strip()] == [x for x in helper_lines if x.strip()])
    strip = lambda s: {k: v for k, v in s.items() if k not in ('hook_node', 'line')}  # noqa: E731
    checks = {
        'normalised_ast_identical': old_norm == new_norm,
        'pre_has_exactly_N_json_default_sites_all_str': len(old_sites) == N_SITES and all(
            s['hook'] == 'str' for s in old_sites),
        'new_has_exactly_N_json_default_sites_all__json_default': len(new_sites) == N_SITES and all(
            s['hook'] == ADDED_DEF for s in new_sites),
        'per_site_enclosing_form_and_other_keywords_unchanged': [
            {**strip(s), 'hook': None} for s in old_sites] == [{**strip(s), 'hook': None} for s in new_sites],
        'non_json_default_keywords_untouched': non_json_defaults(old_src) == non_json_defaults(new_src),
        'exactly_one_added_def__json_default_absent_before': len(added) == 1 and not pre_had,
        'line_diff_only_N_substitutions_plus_helper_block': (substitutions == N_SITES and not other
                                                             and inserted_is_helper),
    }
    return {'checks': checks,
            'normalisation': (f'value of every `default=` keyword of json.dump/json.dumps replaced by Name('
                              f'{PLACEHOLDER!r}) on both sides; the single top-level `def {ADDED_DEF}` removed from '
                              'the new side; ast.dump(include_attributes=False) compared'),
            'first_normalised_difference': first_diff,
            'sites_pre': [{**strip(s), 'line': s['line']} for s in old_sites],
            'sites_new': [{**strip(s), 'line': s['line']} for s in new_sites],
            'non_json_default_keywords': [list(t) for t in non_json_defaults(new_src)],
            'line_diff': {'substituted_lines': substitutions, 'inserted_blocks': inserted, 'other_changes': other},
            'added_helper_source': ast.get_source_segment(new_src, added[0]) if added else None}


# ----------------------------------------------------------------------------------------------------------------------
def _bool_payload():
    a, b = np.float64(0.25), np.float64(0.5)
    t, f = a < b, a > b          # numpy booleans from numpy comparisons (as `relative < threshold` in the module)
    return {'flag_true': t, 'flag_false': f, 'nested': {'objective_convergence': t, 'determinate_at_gt_error_bar': f},
            'list': [t, f], 'plain_float': a}, (t, f)


def _serialise_at_site(site, hook, fname):
    """Serialise with the site's own call form and keywords (evaluated literally) and the given hook."""
    kwargs = {k: ast.literal_eval(v) for k, v in site['other_keywords'].items()}
    payload, _ = _bool_payload()
    if site['form'] == 'dump':
        path = os.path.join(OUT, fname)
        with open(path, 'x') as handle:
            json.dump(payload, handle, default=hook, **kwargs)
        with open(path) as handle:
            return handle.read(), os.path.relpath(path, REPO)
    return json.dumps(payload, default=hook, **kwargs), None


def _strict(back):
    return {'flag_true_is_True': back['flag_true'] is True, 'flag_false_is_False': back['flag_false'] is False,
            'nested_is_True_is_False': (back['nested']['objective_convergence'] is True
                                        and back['nested']['determinate_at_gt_error_bar'] is False),
            'list_strict': back['list'][0] is True and back['list'][1] is False,
            'float_unchanged': back['plain_float'] == 0.25 and type(back['plain_float']) is float}


def v2_v3_controls(old_src, new_src):
    _t, old_sites = json_default_sites(old_src)
    _t, new_sites = json_default_sites(new_src)
    _payload, (t, f) = _bool_payload()
    per_site = []
    for k, (old, new) in enumerate(zip(old_sites, new_sites)):
        new_hook = eval(compile(ast.Expression(new['hook_node']), '<site>', 'eval'), vars(G))  # noqa: S307
        old_hook = eval(compile(ast.Expression(old['hook_node']), '<site>', 'eval'), vars(G))  # noqa: S307
        text_new, path_new = _serialise_at_site(new, new_hook, f'v2_site{k + 1:02d}_line{new["line"]}.json')
        text_old, path_old = _serialise_at_site(old, old_hook, f'v3_site{k + 1:02d}_prehook_line{old["line"]}.json')
        back_new, back_old = json.loads(text_new), json.loads(text_old)
        strict_new, strict_old = _strict(back_new), _strict(back_old)
        per_site.append({
            'site': k + 1, 'line_pre': old['line'], 'line_new': new['line'], 'enclosing': new['enclosing'],
            'form': new['form'], 'other_keywords': new['other_keywords'],
            'new_hook_is_G__json_default': new_hook is G._json_default, 'pre_hook_is_builtin_str': old_hook is str,
            'new_values': [back_new['flag_true'], back_new['flag_false']], 'new_strict': strict_new,
            'pre_values': [back_old['flag_true'], back_old['flag_false']], 'pre_strict': strict_old,
            'written_new': path_new, 'written_pre': path_old})
    direct = os.path.join(OUT, 'v2_direct__atomic_write_json.json')
    G._atomic_write_json(direct, _bool_payload()[0])
    direct_back = json.load(open(direct))
    checks_v2 = {
        'payload_flags_are_numpy_bools_not_python_bools': (type(t).__module__ == 'numpy' and not isinstance(t, bool)
                                                           and type(f).__module__ == 'numpy'),
        'every_site_hook_resolves_to_G__json_default': all(s['new_hook_is_G__json_default'] for s in per_site),
        'every_site_writes_json_booleans_strict_is_True_is_False': all(
            all(s['new_strict'].values()) for s in per_site),
        'G__atomic_write_json_called_directly_strict': all(_strict(direct_back).values()),
        'n_sites': len(per_site) == N_SITES,
    }
    checks_v3 = {
        'every_pre_site_hook_is_builtin_str': all(s['pre_hook_is_builtin_str'] for s in per_site),
        'pre_hook_reproduces_strings_True_False_at_every_site': all(
            s['pre_values'] == ['True', 'False'] for s in per_site),
        'strict_read_fails_on_pre_hook_output_at_every_site': all(
            not s['pre_strict']['flag_true_is_True'] and not s['pre_strict']['flag_false_is_False'] for s in per_site),
        'truthiness_inverts_pre_hook_False': all(bool(s['pre_values'][1]) is True for s in per_site),
    }
    return ({'checks': checks_v2, 'numpy_version': np.__version__,
             'numpy_bool_type': f'{type(t).__module__}.{type(t).__name__}', 'per_site': per_site,
             'direct__atomic_write_json': os.path.relpath(direct, REPO)},
            {'checks': checks_v3, 'note': 'pre hook = the site expression of the pre-change AST (builtin str)'})


# ----------------------------------------------------------------------------------------------------------------------
class _Custom:
    def __str__(self):
        return 'custom-object-str'


def _battery():
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=1.5)
    return {
        'pathlib.PosixPath': pathlib.Path('/tmp/some/dir/file.json'),
        'datetime.datetime': _dt.datetime(2026, 9, 25, 10, 14, 9, 265686, tzinfo=_dt.timezone.utc),
        'datetime.date': _dt.date(2026, 9, 25),
        'numpy.float16': np.float16(0.1), 'numpy.float32': np.float32(0.1), 'numpy.int8': np.int8(-3),
        'numpy.int32': np.int32(7), 'numpy.int64': np.int64(93635360), 'numpy.uint8': np.uint8(200),
        'numpy.ndarray': np.array([1.0, 2.5]), 'set': {3, 1, 2}, 'frozenset': frozenset({'a'}),
        'decimal.Decimal': decimal.Decimal('1.10'), 'complex': complex(1, -2), 'bytes': b'ab',
        'pyomo.ScalarVar': m.x, 'custom_object': _Custom(),
    }


NATIVE = {'bool': True, 'None': None, 'int': 5, 'float': 0.1, 'numpy.float64 (float subclass)': np.float64(0.1),
          'nan_float': float('nan'), 'str': 'True', 'list': [1, 'a'], 'dict': {'k': [False]}}


def _shape(s):
    if s in ('True', 'False'):
        return 'bool-like'
    if re.fullmatch(r'[-+]?\d+', s):
        return 'int-like'
    if re.fullmatch(r'[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?|nan|inf|-inf', s):
        return 'float-like'
    if s.startswith('<') and s.endswith('>'):
        return 'repr-like'
    if s.startswith('{') and s.endswith('}'):
        return 'set/dict-like'
    if s.startswith('[') and s.endswith(']'):
        return 'array-like'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}:\d{2}.*)?', s):
        return 'date-like'
    return None


def _walk(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, key, out)
    elif isinstance(obj, str):
        shape = _shape(obj)
        if shape is not None:
            out[(key, shape)] += 1
            out[('__example__', key, shape)] = obj


def v4_non_bool(new_src):
    _t, sites = json_default_sites(new_src)
    kw_sets = sorted({json.dumps(s['other_keywords'], sort_keys=True) for s in sites})
    battery = _battery()
    per_type = {}
    for name, obj in battery.items():
        texts = {}
        for kws in kw_sets:
            kwargs = {k: ast.literal_eval(v) for k, v in json.loads(kws).items()}
            texts[kws] = (json.dumps({'v': obj, 'l': [obj]}, default=G._json_default, **kwargs)
                          == json.dumps({'v': obj, 'l': [obj]}, default=str, **kwargs))
        per_type[name] = {'hook_equals_str': G._json_default(obj) == str(obj), 'str': str(obj)[:120],
                          'dumps_identical_per_keyword_set': texts}
    native = {}
    for name, obj in NATIVE.items():
        native[name] = all(json.dumps({'v': obj}, default=G._json_default, **{k: ast.literal_eval(v) for k, v in
                                                                             json.loads(kws).items()})
                           == json.dumps({'v': obj}, default=str, **{k: ast.literal_eval(v) for k, v in
                                                                     json.loads(kws).items()})
                           for kws in kw_sets)
    # (b) evidence of the types actually present in committed artifacts written by this module
    pats = [(k, re.compile(r'(^|/)' + k + r'$')) for k in G_WRITTEN_PATTERNS]
    tracked = []
    for p in _git('ls-files', 'data/SRP1/Results').splitlines():
        kind = next((k for k, rx in pats if rx.search(p)), None)
        if kind is not None:
            tracked.append((p, kind))
    by_kind = {}
    n_parsed, parse_errors = 0, []
    for rel, kind in tracked:
        counts = by_kind.setdefault(kind, Counter())
        try:
            with open(os.path.join(REPO, rel)) as handle:
                if rel.endswith('.jsonl'):
                    for line in handle:
                        if line.strip():
                            _walk(json.loads(line), None, counts)
                else:
                    _walk(json.load(handle), None, counts)
            n_parsed += 1
        except Exception as error:  # noqa: BLE001
            parse_errors.append({'path': rel, 'error': f'{type(error).__name__}: {error}'[:200]})
    shapes = {}
    for kind, counts in sorted(by_kind.items()):
        rows = {}
        for key, n in counts.items():
            if key[0] == '__example__':
                continue
            field, shape = key
            rows[f'{field}|{shape}'] = {'n': n, 'example': counts[('__example__', field, shape)]}
        shapes[kind] = dict(sorted(rows.items()))
    checks = {
        'battery_hook_equals_str_every_type': all(v['hook_equals_str'] for v in per_type.values()),
        'battery_dumps_byte_identical_every_type_every_site_keyword_set': all(
            all(v['dumps_identical_per_keyword_set'].values()) for v in per_type.values()),
        'json_native_values_byte_identical': all(native.values()),
        'committed_g_written_artifacts_all_parsed': not parse_errors,
    }
    return {'checks': checks, 'site_keyword_sets': kw_sets, 'battery': per_type, 'native': native,
            'committed_shape_census': {'n_tracked_files': len(tracked), 'n_parsed': n_parsed,
                                       'parse_errors': parse_errors, 'patterns': list(G_WRITTEN_PATTERNS),
                                       'shapes_by_file_kind': shapes,
                                       'caveat': ('a shape match is a candidate hook output, not proof: a genuine '
                                                  'string value can have the same shape')}}


# ----------------------------------------------------------------------------------------------------------------------
def _py_files():
    out = []
    for dirpath, dirs, files in os.walk(REPO):
        rel_dir = os.path.relpath(dirpath, REPO)
        if rel_dir == '.git' or rel_dir.startswith('.git' + os.sep) or rel_dir.startswith(
                os.path.join('.claude', 'worktrees')):
            dirs[:] = []
            continue
        out += [os.path.normpath(os.path.join(rel_dir, f)) for f in files if f.endswith('.py')]
    return sorted(out)


def _changed_functions(old_src, new_src):
    def tops(src):
        return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    o, n = tops(old_src), tops(new_src)
    return {k: (o.get(k), n.get(k)) for k in set(o) | set(n) if o.get(k) != n.get(k)}


def v5_blast_radius(old_src, new_src):
    py = _py_files()
    texts = {}
    for rel in py:
        try:
            texts[rel] = open(os.path.join(REPO, rel), encoding='utf-8', errors='replace').read()
        except OSError:
            continue
    # (a) literal pins of any historical module hash
    commits = _git('log', '--format=%H', '--', MODULE_REL).split()
    hist = {}
    for c in commits:
        try:
            hist[_sha_bytes(_git('show', f'{c}:{MODULE_REL}', binary=True))] = c[:8]
        except subprocess.CalledProcessError:
            pass
    literal_pins = [{'file': rel, 'sha256': h, 'commit': c} for rel, t in texts.items() if rel != os.path.basename(
        __file__) for h, c in hist.items() if h in t or h[:16] in t]
    # (b) every line naming the module file
    naming = []
    for rel, t in texts.items():
        if rel in (MODULE_REL, os.path.basename(__file__)):
            continue
        for i, line in enumerate(t.splitlines(), 1):
            if MODULE_REL in line:
                naming.append({'file': rel, 'line': i, 'text': line.strip()[:200]})
    # (c) string literals whose presence in the module / changed functions flips
    changed = _changed_functions(old_src, new_src)
    importers = [rel for rel, t in texts.items() if 'p515_g_g1_g4_admm_gates' in t and rel != MODULE_REL
                 and rel != os.path.basename(__file__)]
    flips, flip_cache = [], {}
    for rel in importers:
        try:
            tree = ast.parse(texts[rel])
        except SyntaxError as error:
            flips.append({'file': rel, 'parse_error': str(error)})
            continue
        lits = {(n.value, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and 3 <= len(n.value) <= 400}
        for lit, line in sorted(lits):
            if lit not in flip_cache:
                where = []
                if (lit in old_src) != (lit in new_src):
                    where.append('module')
                for fn, (o, n) in changed.items():
                    if (lit in (o or '')) != (lit in (n or '')):
                        where.append(fn)
                flip_cache[lit] = where
            where = flip_cache[lit]
            if where:
                flips.append({'file': rel, 'line': line, 'literal': lit[:200], 'flips_in': where,
                              'present_pre_module': lit in old_src, 'present_new_module': lit in new_src})
    # (d) (file, line, needle) citations of the module
    cite_re = re.compile(r"\(\s*'" + re.escape(MODULE_REL) + r"'\s*,\s*(\d+)\s*,\s*(['\"])(.*?)\2")
    old_lines, new_lines = old_src.splitlines(), new_src.splitlines()
    citations = []
    for rel, t in texts.items():
        for m in cite_re.finditer(t):
            ln, needle = int(m.group(1)), m.group(3)
            citations.append({'file': rel, 'cited_line': ln, 'needle': needle,
                              'verifies_pre': ln <= len(old_lines) and needle in old_lines[ln - 1],
                              'verifies_new': ln <= len(new_lines) and needle in new_lines[ln - 1],
                              'needle_now_at_new_lines': [i for i, x in enumerate(new_lines, 1) if needle in x][:10]})
    # (e) committed JSON artifacts recording the pre-change sha256
    recorded = _git('grep', '-l', PRE_SHA256, '--', '*.json').split()
    # (f) historical resolution
    resolves = _sha_bytes(_git('show', f'{PRE_COMMIT}:{MODULE_REL}', binary=True)) == PRE_SHA256
    checks = {
        'no_py_file_pins_any_historical_module_sha256_literal': literal_pins == [],
        'pre_change_blob_resolves_from_git_to_pinned_sha256': resolves,
    }
    return {'checks': checks, 'n_py_files_searched': len(texts), 'search_scope': ('every *.py under the repository '
            'root excluding .git/ and .claude/worktrees/, tracked or not'),
            'historical_module_sha256': hist, 'literal_pins': literal_pins, 'lines_naming_module': naming,
            'changed_top_level_functions': sorted(changed), 'importers_scanned_for_literal_flips': len(importers),
            'literal_presence_flips': flips, 'line_citations': citations,
            'committed_json_recording_pre_sha256': recorded,
            'module_sha256_pre': PRE_SHA256, 'module_sha256_new': _sha(MODULE_REL)}


# ----------------------------------------------------------------------------------------------------------------------
def main():
    started = time.time()
    if os.path.exists(OUT):
        raise SystemExit(f'REFUSED: output root exists (write-once): {OUT}')
    os.makedirs(OUT)
    new_src = open(os.path.join(REPO, MODULE_REL)).read()
    if os.path.abspath(G.__file__) != os.path.join(REPO, MODULE_REL):
        raise SystemExit(f'REFUSED: imported module {G.__file__} is not {MODULE_REL} in this checkout')
    old_src = pre_source()
    sections, errors = {}, {}
    for name, fn in (('V1_ast_equality', lambda: v1_ast(old_src, new_src)),
                     ('V2_V3_controls', lambda: v2_v3_controls(old_src, new_src)),
                     ('V4_non_bool_unchanged', lambda: v4_non_bool(new_src)),
                     ('V5_pin_blast_radius', lambda: v5_blast_radius(old_src, new_src))):
        try:
            result = fn()
            if name == 'V2_V3_controls':
                sections['V2_positive_control'], sections['V3_negative_control'] = result
            else:
                sections[name] = result
        except Exception as error:  # noqa: BLE001
            errors[name] = f'{type(error).__name__}: {error}\n{traceback.format_exc()}'
            print(f'[W76] {name} ERROR {errors[name]}', flush=True)
    failing = sorted(f'{s}.{k}' for s, v in sections.items() for k, ok in v['checks'].items() if ok is not True)
    guard_failures = GUARD.verify(0)
    payload = {
        'task': 'P5.15 W76 item 2 -- p515_g_g1_g4_admm_gates.py _json_default, confined-change zero-solve checks',
        'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').strip(),
        'git_status_module': _git('status', '--porcelain', '--', MODULE_REL).strip(),
        'module': MODULE_REL, 'module_sha256_new': _sha(MODULE_REL), 'module_sha256_pre': PRE_SHA256,
        'pre_commit': PRE_COMMIT, 'script_sha256': _sha(os.path.abspath(__file__)), 'sections': sections,
        'errors': errors, 'failing': failing, 'pass': (not failing and not errors and not guard_failures),
        'guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_s': time.time() - started}
    with open(OUT_JSON, 'x') as handle:
        json.dump(payload, handle, indent=1, default=G._json_default)
    for s, v in sections.items():
        for k, ok in v['checks'].items():
            print(f'[W76] {s}: {k}: {"PASS" if ok is True else "FAIL"}', flush=True)
    print(f'[W76] failing={failing} errors={sorted(errors)} guard {dict(GUARD.counts)} verify(0) {guard_failures}',
          flush=True)
    print(f"[W76] {'PASS' if payload['pass'] else 'FAIL'} -> {os.path.relpath(OUT_JSON, REPO)}; "
          f"wall {payload['wall_s']:.1f}s", flush=True)
    GUARD.uninstall()
    sys.exit(0 if payload['pass'] else 1)


def manifest():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
    m = {}
    for dirpath, _dirs, files in os.walk(OUT):
        for fname in sorted(files):
            fpath = os.path.join(dirpath, fname)
            m[os.path.relpath(fpath, REPO)] = _sha(fpath)
    for rel in (LAUNCH_LOG_REL, os.path.basename(os.path.abspath(__file__)), MODULE_REL):
        m[rel] = _sha(rel)
    with open(OUT_MANIFEST, 'x') as handle:
        json.dump(m, handle, indent=1)
    failures = GUARD.verify(0)
    print(f'[W76] wrote {os.path.relpath(OUT_MANIFEST, REPO)}: {len(m)} entries; guard verify(0) {failures}',
          flush=True)
    GUARD.uninstall()
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--manifest']:
        manifest()
    elif sys.argv[1:]:
        raise SystemExit(f'unknown arguments: {sys.argv[1:]}')
    else:
        main()
