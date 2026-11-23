import sys, json, os
sys.path.insert(0, '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')
import p515_s56_coupling_gates as G
a = G._abs('data/SRP1/Results/P515S56/coupling_option_v1/campaign_s56cpl_gd_pre/evals/ca8927e75d628bd1_y2025__n7_p0_25_e1')
b = G._abs(G.SELFTEST['same_A'])
pa, pb = G.per_cycle_files(a), G.per_cycle_files(b)
out = {'rows': {k: (len(pa[k]), len(pb[k])) for k in pa}}
g1 = {}
for k in pa:
    n = len(pa[k])
    d, _ = G.deep_diff(pa[k], pb[k][:n])
    g1[k] = {'n_compared_rows': n, 'n_differences': len(d), 'first': d[:3]}
out['G1_prefix'] = g1
ra = G._read_jsonl(os.path.join(a, G.G2_RECORDS_FILE)); rb = G._read_jsonl(os.path.join(b, G.G2_RECORDS_FILE))
d, ex = G.deep_diff(ra, rb[:len(ra)], excluded=tuple(G.G2_RECORDS_EXCLUDED))
out['G2_records_prefix'] = {'n_a': len(ra), 'n_b': len(rb), 'n_differences': len(d), 'first': d[:3]}
ea = G._load(os.path.join(a, 'evaluation_record.json')); eb = G._load(os.path.join(b, 'evaluation_record.json'))
la = G._abs(os.path.join(G.P56A_WORK_DIR_REL, ea['working_dir_ids']['run'], 'logs'))
lb = G._abs(os.path.join(G.P56A_WORK_DIR_REL, eb['working_dir_ids']['run'], 'logs'))
fa, fb = sorted(os.listdir(la)), sorted(os.listdir(lb))
nb = nd = 0; firstd = []
for name in sorted(set(fa) & set(fb)):
    xa, xb = G.ipopt_final_blocks(os.path.join(la, name)), G.ipopt_final_blocks(os.path.join(lb, name))
    for i, blk in enumerate(xa['final_blocks']):
        nb += 1
        if i >= len(xb['final_blocks']) or blk != xb['final_blocks'][i]:
            nd += 1; firstd.append((name, i))
out['ipopt_prefix'] = {'files_a': len(fa), 'files_b': len(fb), 'same_file_names': fa == fb, 'n_blocks_a_compared': nb,
                       'n_differences': nd, 'first': firstd[:3]}
out['guard_zero'] = G.guard_report()['verified_zero']
print(json.dumps(out, indent=1, default=str))
