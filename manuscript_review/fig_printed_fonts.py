"""Printed font family and size of every text run in matplotlib PDFs, at their inclusion width in the manuscript.

Zero solves; reads PDFs only. For each file: page width (pt), inclusion width (fraction x \\textwidth = 390 pt,
elsarticle 'preprint,12pt'), scale = inclusion width / page width; then each text run's font (/BaseFont of the
resource it selects) and printed size = Tf size x scale. Text runs are decoded with the operator parser of
fig_storage_profiles_compare.py.

Usage: <canonical python> manuscript_review/fig_printed_fonts.py --json OUT FILE:FRACTION [FILE:FRACTION ...]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_storage_profiles_compare as C  # noqa: E402

TEXTWIDTH_PT = 390.0


def font_names(b):
    """Map the font resource names (/F1 ...) to their /BaseFont (the /Font dictionary may be direct or indirect)."""
    objs = {m.group(1).decode(): m.group(2) for m in re.finditer(rb'(?:^|\n)(\d+)\s+0\s+obj\s*(.*?)endobj', b, re.S)}
    out = {}
    for m in re.finditer(rb'/Font\s*(<<.*?>>|\d+\s+0\s+R)', b, re.S):
        d = m.group(1)
        if d.endswith(b'R'):
            d = objs.get(d.split()[0].decode(), b'')
        for name, ref in re.findall(rb'/([A-Za-z0-9]+)\s+(\d+)\s+0\s+R', d):
            bf = re.search(rb'/BaseFont\s*/([A-Za-z0-9+\-]+)', objs.get(ref.decode(), b''))
            if bf:
                out[name.decode()] = bf.group(1).decode()
    return out


def report(path, frac):
    b = open(path, 'rb').read()
    page, ops = C.ops(path, 0.001)
    w = float(page[0][2]) - float(page[0][0])
    scale = frac * TEXTWIDTH_PT / w
    names = font_names(b)
    runs = []
    for o in ops:
        if o[0] == 'text' and o[3]:
            res, size = o[3]
            runs.append({'text': o[2], 'font': names.get(res.lstrip('/'), res), 'printed_pt': round(float(size) * scale, 2)})
    sizes = Counter((r['font'], r['printed_pt']) for r in runs)
    return {'file': path, 'page_width_pt': w, 'inclusion_fraction': frac, 'scale': scale,
            'families': sorted(set(r['font'] for r in runs)),
            'printed_sizes': sorted({r['printed_pt'] for r in runs}),
            'by_font_and_size': {f'{f} {s}': n for (f, s), n in sorted(sizes.items())}, 'runs': runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    out = []
    for spec in a.files:
        path, frac = spec.rsplit(':', 1)
        r = report(path, float(frac))
        out.append(r)
        print(f"{os.path.basename(path)} @ {frac} textwidth (scale {r['scale']:.4f}): {r['families']} "
              f"sizes {r['printed_sizes']}")
    with open(a.json, 'x', encoding='utf-8') as h:
        json.dump(out, h, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
