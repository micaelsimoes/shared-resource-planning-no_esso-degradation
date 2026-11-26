"""Pin check f6c23977 (zero solves, no model build): compare the HEAD NL-digest run with the spec v1 reference
(w183_nl_post_w183c_off, production 6f43479e), block by block, using the same equality definition as W183c gate 2
(`p515_s53_w183c_nrf_gates._block_files_equal`, reproduced field for field below), plus three independent checks:
(a) the gz sha256 of every NL and per-block item file in the two manifests (path re-rooted), (b) the decompressed
sha256 of every committed ESSO NL gz recomputed from the files, (c) the determinism child of both runs.
Usage: python pincheck_compare.py <ref_dir> <post_dir> <out_json>"""
import gzip
import hashlib
import json
import os
import sys

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
ALL_CONFIGS = [f'{i}_{m}_{f}' for i in ('x0', 'unit') for m in ('A', 'N') for f in ('1x', '2x')]
BUILDERS = ('sequential', 'parallel')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_gz(p):
    with gzip.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(d):
    raw = os.path.join(d, 'w183_nl_digest.json')
    gz = raw + '.gz'
    if os.path.isfile(gz):
        with gzip.open(gz, 'rt') as f:
            return json.load(f), {'source': os.path.relpath(gz, REPO), 'sha256': sha_file(gz)}
    with open(raw) as f:
        return json.load(f), {'source': os.path.relpath(raw, REPO), 'sha256': sha_file(raw)}


def files_equal(a, b):
    files = {'plain': (a['nl']['plain'] or {}).get('sha256') == (b['nl']['plain'] or {}).get('sha256')}
    for f in ('nl', 'row', 'col'):
        files[f'symbolic_{f}'] = ((a['nl']['symbolic'] or {}).get(f, {}).get('sha256')
                                  == (b['nl']['symbolic'] or {}).get(f, {}).get('sha256'))
    files['plain_sha256_not_null'] = (a['nl']['plain'] or {}).get('sha256') is not None
    files['block_digest'] = a['sha256'] == b['sha256']
    files['component_digests'] = a['components'] == b['components']
    return files


def compare_records(ref, post, configs):
    per, n, n_id, esso = {}, 0, 0, {}
    for c in configs:
        per[c] = {}
        for b in BUILDERS:
            ra, rb = ref['records'][c][b]['blocks'], post['records'][c][b]['blocks']
            diffs = []
            for key in sorted(set(ra) | set(rb)):
                if key not in ra or key not in rb:
                    diffs.append({'block': key, 'missing_in': 'reference' if key not in ra else 'post'})
                    continue
                fe = files_equal(ra[key], rb[key])
                n += 1
                if all(fe.values()):
                    n_id += 1
                else:
                    comp_diff = sorted(k for k in set(ra[key]['components']) | set(rb[key]['components'])
                                       if ra[key]['components'].get(k) != rb[key]['components'].get(k))
                    diffs.append({'block': key, 'files': fe, 'differing_components': comp_diff})
                if key.startswith('ESSO'):
                    esso[f'{c}/{b}/{key}'] = {
                        'plain_sha256_reference': ra[key]['nl']['plain']['sha256'],
                        'plain_sha256_post': rb[key]['nl']['plain']['sha256'],
                        'all_equal': all(fe.values())}
            per[c][b] = {'n_blocks_reference': len(ra), 'n_blocks_post': len(rb),
                         'n_tso': sum(1 for k in rb if k.startswith('TSO')),
                         'n_dso': sum(1 for k in rb if k.startswith('DSO')),
                         'n_esso': sum(1 for k in rb if k.startswith('ESSO')),
                         'n_differences': len(diffs), 'differences': diffs}
    return per, n, n_id, esso


def manifest_compare(ref_dir, post_dir):
    with open(os.path.join(ref_dir, 'manifest_sha256.json')) as f:
        mr = json.load(f)
    with open(os.path.join(post_dir, 'manifest_sha256.json')) as f:
        mp = json.load(f)
    rr, pr = os.path.relpath(ref_dir, REPO) + '/', os.path.relpath(post_dir, REPO) + '/'
    strip = lambda m, r: {k[len(r):]: v for k, v in m.items()}
    a, b = strip(mr, rr), strip(mp, pr)
    payload = lambda m: {k: v for k, v in m.items() if k.endswith('.gz') and not k.endswith('w183_nl_digest.json.gz')}
    pa, pb = payload(a), payload(b)
    differing = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    return {'n_reference_entries': len(a), 'n_post_entries': len(b), 'n_payload_gz_reference': len(pa),
            'n_payload_gz_post': len(pb), 'n_payload_gz_differing_or_missing': len(differing),
            'payload_gz_differing_or_missing': differing[:50],
            'non_payload_entries_reference': sorted(set(a) - set(pa)), 'non_payload_entries_post': sorted(set(b) - set(pb))}


def esso_gz_recompute(ref_dir, post_dir, ref, post):
    out, ok = {}, True
    for sub, rj, pj in (('', ref, post), ('determinism_child/', None, None)):
        for c in ALL_CONFIGS:
            for node in (5, 7, 9):
                for ext, field in (('nl', ('plain',)), ('sym.nl', ('symbolic', 'nl')), ('sym.row', ('symbolic', 'row')),
                                   ('sym.col', ('symbolic', 'col'))):
                    rel = f'{sub}nl/{c}/sequential/ESSO_node{node}.{ext}.gz'
                    pr, pp = os.path.join(ref_dir, rel), os.path.join(post_dir, rel)
                    if not (os.path.isfile(pr) or os.path.isfile(pp)):
                        continue
                    hr = sha_gz(pr) if os.path.isfile(pr) else None
                    hp = sha_gz(pp) if os.path.isfile(pp) else None
                    out[rel] = {'reference': hr, 'post': hp, 'equal': hr is not None and hr == hp}
                    ok &= out[rel]['equal']
    return out, ok


def main():
    ref_dir, post_dir, out_json = (os.path.abspath(p) for p in sys.argv[1:4])
    ref, ref_src = load_json(ref_dir)
    post, post_src = load_json(post_dir)
    per, n, n_id, esso = compare_records(ref, post, ALL_CONFIGS)
    keys = {c: {'reference': ref['records'][c]['eval_key_base'], 'post': post['records'][c]['eval_key_base'],
                'equal': ref['records'][c]['eval_key_base'] == post['records'][c]['eval_key_base']} for c in ALL_CONFIGS}
    rc, pc = load_json(os.path.join(ref_dir, 'determinism_child'))[0], load_json(os.path.join(post_dir, 'determinism_child'))[0]
    det_cfgs = sorted(set(rc['records']) & set(pc['records']))
    dper, dn, dn_id, _ = compare_records(rc, pc, det_cfgs)
    man = manifest_compare(ref_dir, post_dir)
    man_child = manifest_compare(os.path.join(ref_dir, 'determinism_child'), os.path.join(post_dir, 'determinism_child'))
    gzr, gz_ok = esso_gz_recompute(ref_dir, post_dir, ref, post)
    prod = lambda j: (j.get('production') or {}).get('modules_sha256') or {}
    prod_diff = {k: {'reference': prod(ref).get(k), 'post': prod(post).get(k)}
                 for k in sorted(set(prod(ref)) | set(prod(post))) if prod(ref).get(k) != prod(post).get(k)}
    inp = lambda j: (j.get('inputs') or {}).get('sha256') or {}
    inp_diff = {k: {'reference': inp(ref).get(k), 'post': inp(post).get(k)}
                for k in sorted(set(inp(ref)) | set(inp(post))) if inp(ref).get(k) != inp(post).get(k)}
    result = {
        'definition': __doc__, 'reference': ref_src, 'post': post_src,
        'reference_git_head_at_run': ref.get('git_head_at_run'), 'post_git_head_at_run': post.get('git_head_at_run'),
        'reference_expected_production_commit': (ref.get('production') or {}).get('expected_commit'),
        'post_expected_production_commit': (post.get('production') or {}).get('expected_commit'),
        'production_modules_differing': prod_diff, 'inputs_differing': inp_diff,
        'n_inputs_reference': len(inp(ref)), 'n_inputs_post': len(inp(post)),
        'post_all_ok': post.get('all_ok'), 'post_checks': post.get('checks'),
        'post_solve_profile_guard': post.get('solve_profile_guard'),
        'post_child_solve_profile_guard': pc.get('solve_profile_guard'),
        'n_blocks_compared': n, 'n_blocks_identical': n_id, 'expected_blocks': len(ALL_CONFIGS) * (51 + 36),
        'per_config': per, 'eval_keys': keys, 'esso_blocks': esso,
        'determinism_child': {'configs': det_cfgs, 'n_blocks_compared': dn, 'n_blocks_identical': dn_id, 'per_config': dper},
        'manifest_gz_payload': man, 'manifest_gz_payload_child': man_child,
        'esso_nl_gz_recomputed': gzr, 'esso_nl_gz_recomputed_all_equal': gz_ok,
    }
    result['verdict_all_identical'] = bool(
        n == n_id == result['expected_blocks'] and all(v['equal'] for v in keys.values()) and dn == dn_id
        and man['n_payload_gz_differing_or_missing'] == 0 and man_child['n_payload_gz_differing_or_missing'] == 0
        and gz_ok and post.get('all_ok') is True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(json.dumps({k: result[k] for k in ('n_blocks_compared', 'n_blocks_identical', 'expected_blocks',
                                             'verdict_all_identical', 'production_modules_differing', 'inputs_differing',
                                             'n_inputs_reference', 'n_inputs_post',
                                             'esso_nl_gz_recomputed_all_equal')}, indent=1))
    print('child', dn, dn_id, det_cfgs)
    print('manifest', {k: v for k, v in man.items() if k != 'payload_gz_differing_or_missing'})
    print('manifest child', {k: v for k, v in man_child.items() if k != 'payload_gz_differing_or_missing'})
    print('keys equal', all(v['equal'] for v in keys.values()))


if __name__ == '__main__':
    main()
