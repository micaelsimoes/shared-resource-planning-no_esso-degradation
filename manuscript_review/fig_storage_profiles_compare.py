"""Operator-level comparison of two matplotlib PDFs (zero solves): what changed between an old and a new figure file.

Each page content stream is decompressed and tokenised into drawing operations: paths (with the stroke colour,
line width and dash pattern in force), marker placements, text runs (decoded from matplotlib's two-byte Unicode codes, with
their position and size). Numbers are compared at --tol points. The report lists every operation present in one file
and not the other, so 'only the label text differs' is a checked statement, not an assertion.

Usage: <canonical python> manuscript_review/fig_storage_profiles_compare.py OLD.pdf NEW.pdf [--tol 0.01] [--json OUT]
                                                                          [--ignore-dash]
Exit 0 always; the JSON/console report states the differences.
"""
import argparse
import json
import re
import zlib


def _streams(b):
    out = []
    for m in re.finditer(rb'<<(.*?)>>\s*stream\r?\n(.*?)\r?\nendstream', b, re.S):
        d = m.group(2)
        if b'/FlateDecode' in m.group(1):
            try:
                d = zlib.decompress(d)
            except zlib.error:
                continue
        out.append((m.group(1), d))
    return out


def _tounicode(b):
    """matplotlib's Type 42 text uses two-byte codes equal to the Unicode code points; no map is needed."""
    return {}


def _decode_tj(arr, cmap):
    s = ''
    for m in re.finditer(rb'\((.*?)(?<!\\)\)', arr, re.S):
        raw = re.sub(rb'\\([()\\])', rb'\1', m.group(1))
        raw = re.sub(rb'\\(\d{3})', lambda mm: bytes([int(mm.group(1), 8)]), raw)
        for i in range(0, len(raw) - 1, 2):
            code = raw[i] * 256 + raw[i + 1]
            s += cmap.get(code, chr(code))
    return s


def ops(path, tol, ignore_dash=False):
    b = open(path, 'rb').read()
    cmap = _tounicode(b)
    content = [d for _h, d in _streams(b) if b' Tf' in d or b' re W n' in d]
    t = max(content, key=len)
    q = lambda v: round(float(v) / tol) * tol   # noqa: E731
    fmt = lambda v: f'{q(v):.3f}'                 # noqa: E731
    state = {'w': None, 'RG': None, 'G': None, 'd': '[]'}
    res, path_pts = [], []
    cm = None
    toks = re.findall(rb'\[[^\]]*\]\s*TJ|\[[^\]]*\]\s*[-0-9.]+\s*d|/[A-Za-z0-9]+|[-0-9.]+|[A-Za-z*]+', t)
    stack = []
    for tk in toks:
        if tk.endswith(b'TJ'):
            res.append(('text', cm, _decode_tj(tk, cmap), state.get('Tf')))
            continue
        if tk.endswith(b'd') and tk.startswith(b'['):
            state['d'] = re.sub(rb'\s+', b' ', tk[:tk.rindex(b']') + 1]).decode().replace('[ ]', '[]')
            continue
        if re.fullmatch(rb'[-0-9.]+', tk) or tk.startswith(b'/'):
            stack.append(tk.decode())
            continue
        op = tk.decode()
        if op == 'w':
            state['w'] = fmt(stack[-1])
        elif op == 'RG':
            state['RG'] = tuple(fmt(v) for v in stack[-3:])
        elif op == 'G':
            state['RG'] = ('gray', fmt(stack[-1]))
        elif op == 'Tf':
            state['Tf'] = (stack[-2], fmt(stack[-1]))
        elif op == 'cm':
            cm = tuple(fmt(v) for v in stack[-6:])
            if stack and len(stack) >= 7 and stack[-7].startswith('/'):
                pass
        elif op == 'Do':
            res.append(('marker', cm, stack[-1], state['RG'], state['w']))
        elif op in ('m', 'l'):
            path_pts.append((fmt(stack[-2]), fmt(stack[-1])))
        elif op == 'c':
            path_pts.append(tuple(fmt(v) for v in stack[-6:]))
        elif op in ('S', 'B', 'f', 'b', 'B*', 'f*'):
            if path_pts:
                res.append(('path', op, state['RG'], state['w'], '*' if ignore_dash else state['d'], tuple(path_pts)))
            path_pts = []
        elif op == 'n':
            path_pts = []
        elif op == 're':
            path_pts.append(('re',) + tuple(fmt(v) for v in stack[-4:]))
        if op not in ('BT', 'ET', 'q', 'Q', 'Td'):
            stack = [] if op != 'cm' else stack
        if op in ('Q',):
            stack = []
    page = re.search(rb'/MediaBox\s*\[([^\]]*)\]', b).group(1).decode().split()
    return [tuple(fmt(v) for v in page)], res


def compare(old, new, tol, ignore_dash=False):
    po, a = ops(old, tol, ignore_dash)
    pn, b = ops(new, tol, ignore_dash)
    from collections import Counter
    ca, cb = Counter(map(repr, a)), Counter(map(repr, b))
    only_old = sorted((ca - cb).elements())
    only_new = sorted((cb - ca).elements())
    kinds = Counter(x[0] for x in a)
    return {'old': old, 'new': new, 'tol_pt': tol, 'ignore_dash': ignore_dash, 'page_old': po, 'page_new': pn, 'page_equal': po == pn,
            'n_ops_old': len(a), 'n_ops_new': len(b), 'kinds_old': dict(kinds),
            'paths_markers_equal': [x for x in only_old if not x.startswith("('text'")] == [] and
            [x for x in only_new if not x.startswith("('text'")] == [],
            'only_in_old': only_old, 'only_in_new': only_new,
            'texts_old': [x[2] for x in a if x[0] == 'text'], 'texts_new': [x[2] for x in b if x[0] == 'text']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--tol', type=float, default=0.01)
    ap.add_argument('--json', default=None)
    ap.add_argument('--ignore-dash', action='store_true', help='compare paths without their dash pattern')
    a = ap.parse_args()
    r = compare(a.old, a.new, a.tol, a.ignore_dash)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as h:
            json.dump(r, h, indent=1, ensure_ascii=False)
    print(f"{a.new}: page equal {r['page_equal']}; ops {r['n_ops_old']} -> {r['n_ops_new']}; paths+markers equal "
          f"{r['paths_markers_equal']}; only-old {len(r['only_in_old'])}, only-new {len(r['only_in_new'])}")
    for x in r['only_in_old'][:12]:
        print('  - ', x[:220])
    for x in r['only_in_new'][:12]:
        print('  + ', x[:220])


if __name__ == '__main__':
    main()
