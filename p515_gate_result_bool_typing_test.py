"""Repository-wide boolean-typing test for gate-result JSON (P5.15 Addendum 52, W100). ZERO SOLVES (stdlib +
`gate_result_io` only; nothing here imports pyomo or a solver).

WHAT IT ASSERTS. Every *.json and *.jsonl file under `data/` (the results tree -- a superset of the gate-result
artifacts; tracked AND untracked files present in the working tree, since a new artifact is untracked until it is
committed and that is when a recurrence must be caught) carries NO string flag, unless the file is on the hash-pinned
list of known historical artifacts (`gate_result_bool_typing_historical.json`) with the SAME sha256.

FLAG DEFINITION (by value; shared with the writer -- `gate_result_io.is_string_flag`). A string flag is any JSON value
(object member value or array element, any depth, any document of a .json file or any line of a .jsonl file; object
KEYS excluded) that is a string whose `strip().lower()` is "true" or "false". A real boolean is JSON `true`/`false`; its
text in any spelling is a string flag. The definition is by value, not by field name, so it catches every field family
the defect has hit -- `objective_convergence`, `determinate_at_gt_error_bar` (W75), `pass` (hull_polish gate,
post_certification.json), `reconciles_to_net` / `step_qualifies` (W98 hooks, G14) -- and any NEW field name.
The `field` reported for a value is its innermost object key (array elements inherit the parent key); keys that are
Pyomo index reprs (starting with "(", a digit or "-") are reported as "<index-key>" (the W75 convention).

VERDICT RULES (each is a FAIL):
  F1 new_string_flag     a file with >= 1 string flag that is not listed (a new file is judged fresh)
  F2 hash_changed        a listed file whose sha256 differs from the pinned sha256 (judged fresh, flags or not)
  F3 listed_tracked_missing   a listed file recorded as git-tracked that is absent from the working tree
  F4 registry_inconsistent    a listed file whose sha256 matches but whose per-field counts differ from the list
                              (the definition or the scanner changed -- the list must be re-derived, not patched)
  F5 registry_invalid    the list does not parse, or an entry lacks path / sha256 / tracked / commit / fields
A listed file recorded as UNTRACKED and absent is reported (`listed_untracked_absent`), not failed: untracked files
are machine-local.

SCAN. Every candidate file is read in full, lower-cased and searched for "true" / "false" followed by optional
whitespace (raw, non-ASCII, or a JSON escape) and a closing quote -- the raw form every string flag must have, so the
prefilter can only over-select. Candidates are parsed (whole-file JSON; else JSON Lines; a line that does not parse
is scanned token by token, a token followed by ":" being a key). Every listed file is hashed whether or not it is a
candidate (F2 must see a change that REMOVED the flags too).

COMMAND (repo root, canonical interpreter, attached, both streams):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \
      [--registry gate_result_bool_typing_historical.json] [--out <new write-once .json>]
Exit 0 = PASS, 1 = FAIL, 2 = error.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gate_result_io as GRIO  # noqa: E402

DEFAULT_REGISTRY = os.path.join(REPO, 'gate_result_bool_typing_historical.json')
DATA_ROOT_REL = 'data'
SUFFIXES = ('.json', '.jsonl')
REGISTRY_SCHEMA = 'gate_result_bool_typing_historical/v1'
REQUIRED_ENTRY_KEYS = ('path', 'sha256', 'tracked', 'commit', 'fields')
INDEX_KEY = re.compile(r'^[\(\d\-]')
_TAIL = rb'(?:[\s\x80-\xff]|\\[bfnrt/]|\\u[0-9a-f]{4})*"'
PREFILTER = (re.compile(rb'true' + _TAIL), re.compile(rb'false' + _TAIL))
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def list_files(repo=REPO):
    out = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(repo, DATA_ROOT_REL)):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(SUFFIXES):
                out.append(os.path.relpath(os.path.join(dirpath, name), repo))
    return out


def tracked_files(repo=REPO):
    text = subprocess.run(['git', 'ls-files', '-z', '--', DATA_ROOT_REL], cwd=repo, capture_output=True,
                          check=True).stdout.decode('utf-8', 'surrogateescape')
    return {p for p in text.split('\0') if p}


def field_label(field):
    if field is None:
        return '<top>'
    return '<index-key>' if INDEX_KEY.match(field) else field


def is_candidate(raw):
    low = raw.lower()
    return any(p.search(low) for p in PREFILTER)


def _raw_token_flags(text):
    """String flags in text that does not parse as JSON: every JSON string token, skipping keys (a token followed,
    after whitespace, by ':')."""
    flags = []
    for m in _TOKEN.finditer(text):
        rest = text[m.end():m.end() + 64].lstrip()
        if rest.startswith(':'):
            continue
        try:
            value = json.loads(m.group(0))
        except ValueError:
            continue
        if GRIO.is_string_flag(value):
            flags.append((f'@char{m.start()}', '<unparsed>', value))
    return flags


def scan_bytes(raw):
    """(parse_mode, [(json_path, field, value), ...], n_unparsed_lines) for one file's bytes."""
    text = raw.decode('utf-8', 'replace')
    try:
        doc = json.loads(text)
        return 'json', GRIO.find_string_flags(doc), 0
    except ValueError:
        pass
    flags, bad = [], 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            bad += 1
            flags.extend((f'line{lineno}{p}', f, v) for p, f, v in _raw_token_flags(line))
            continue
        flags.extend((f'line{lineno}{p[1:]}' if p.startswith('$') else p, f, v)
                     for p, f, v in GRIO.find_string_flags(doc))
    return 'jsonl', flags, bad


def field_counts(flags):
    counts = {}
    for _path, field, value in flags:
        per = counts.setdefault(field_label(field), Counter())
        per[value] += 1
    return {f: dict(sorted(c.items())) for f, c in sorted(counts.items())}


def scan_file(rel, repo=REPO):
    with open(os.path.join(repo, rel), 'rb') as handle:
        raw = handle.read()
    if not is_candidate(raw):
        return {'candidate': False, 'flags': [], 'sha256': None}
    mode, flags, bad = scan_bytes(raw)
    return {'candidate': True, 'parse': mode, 'unparsed_lines': bad, 'flags': flags,
            'sha256': hashlib.sha256(raw).hexdigest() if flags else None}


def load_registry(path):
    with open(path) as handle:
        reg = json.load(handle)
    problems = []
    if reg.get('schema') != REGISTRY_SCHEMA:
        problems.append(f"schema {reg.get('schema')!r} != {REGISTRY_SCHEMA!r}")
    entries = reg.get('entries')
    if not isinstance(entries, list):
        problems.append('entries is not a list')
        entries = []
    seen = set()
    for i, e in enumerate(entries):
        missing = [k for k in REQUIRED_ENTRY_KEYS if k not in e]
        if missing:
            problems.append(f'entry {i}: missing {missing}')
        elif not (isinstance(e['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', e['sha256'])):
            problems.append(f"entry {i}: sha256 is not 64 hex digits: {e['sha256']!r}")
        elif e['tracked'] is True and not (isinstance(e['commit'], str) and re.fullmatch(r'[0-9a-f]{40}', e['commit'])):
            problems.append(f"entry {i}: tracked but commit is not a full hash: {e['commit']!r}")
        if e.get('path') in seen:
            problems.append(f"entry {i}: duplicate path {e.get('path')}")
        seen.add(e.get('path'))
    return reg, {e['path']: e for e in entries if 'path' in e}, problems


def evaluate(registry_path=DEFAULT_REGISTRY, repo=REPO, progress=False):
    t0 = time.time()
    try:
        reg, listed, problems = load_registry(registry_path)
    except (OSError, ValueError) as error:
        reg, listed, problems = None, {}, [f'registry unreadable: {type(error).__name__}: {error}']
    files = list_files(repo)
    tracked = tracked_files(repo)
    present = set(files)
    failures = {'F1_new_string_flag': [], 'F2_hash_changed': [], 'F3_listed_tracked_missing': [],
                'F4_registry_inconsistent': [], 'F5_registry_invalid': list(problems)}
    known, absent_untracked = [], []
    n_candidates, n_unparsed_files = 0, 0
    for i, rel in enumerate(files):
        if progress and i % 20000 == 0:
            print(f'  scanned {i}/{len(files)} ({time.time() - t0:.0f} s)', flush=True)
        entry = listed.get(rel)
        res = scan_file(rel, repo)
        n_candidates += res['candidate']
        n_unparsed_files += bool(res.get('unparsed_lines'))
        if entry is None:
            if res['flags']:
                failures['F1_new_string_flag'].append({
                    'path': rel, 'tracked': rel in tracked, 'sha256': res['sha256'],
                    'fields': field_counts(res['flags']),
                    'examples': [[p, f, repr(v)] for p, f, v in res['flags'][:5]]})
            continue
        digest = res['sha256'] or sha256_file(os.path.join(repo, rel))
        if digest != entry['sha256']:
            failures['F2_hash_changed'].append({'path': rel, 'pinned_sha256': entry['sha256'], 'sha256': digest,
                                                'string_flags_now': field_counts(res['flags'])})
            continue
        counts = field_counts(res['flags'])
        if counts != entry['fields']:
            failures['F4_registry_inconsistent'].append({'path': rel, 'listed': entry['fields'], 'scanned': counts})
            continue
        known.append(rel)
    for rel, entry in sorted(listed.items()):
        if rel in present:
            continue
        if entry.get('tracked') is True:
            failures['F3_listed_tracked_missing'].append(rel)
        else:
            absent_untracked.append(rel)
    ok = not any(failures.values())
    return {
        'verdict': 'PASS' if ok else 'FAIL',
        'ok': ok,
        'registry': os.path.relpath(registry_path, repo) if os.path.isabs(registry_path) else registry_path,
        'registry_sha256': sha256_file(registry_path) if os.path.exists(registry_path) else None,
        'registry_entries': len(listed),
        'scope': {'root': DATA_ROOT_REL, 'suffixes': list(SUFFIXES), 'files': len(files),
                  'tracked_files': sum(1 for f in files if f in tracked), 'candidates': n_candidates,
                  'files_with_unparsed_lines': n_unparsed_files},
        'known_historical_matched': len(known),
        'listed_untracked_absent': absent_untracked,
        'failures': failures,
        'failure_counts': {k: len(v) for k, v in failures.items()},
        'wall_s': time.time() - t0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--registry', default=DEFAULT_REGISTRY)
    parser.add_argument('--out', default=None, help='write-once JSON result (through gate_result_io)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.registry, progress=not args.quiet)
    except Exception as error:  # noqa: BLE001
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 2
    if not args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k != 'failures'}, indent=1))
        for name, items in result['failures'].items():
            for item in items[:20]:
                print(f'{name}: {json.dumps(item)[:400]}')
        print(f"VERDICT: {result['verdict']}", flush=True)
    if args.out:
        if os.path.exists(args.out):
            print(f'ERROR: refusing to overwrite {args.out}', file=sys.stderr)
            return 2
        with open(args.out, 'x') as handle:
            GRIO.dump(result, handle, indent=1)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
