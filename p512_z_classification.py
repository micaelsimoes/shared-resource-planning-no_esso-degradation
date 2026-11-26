"""P5.12-Z: re-derive the W/X/Y classification numbers from primary artifacts.

Frozen spec: data/SRP1/Results/P512Z/frozen_formula_spec.json
No solve. Guards armed and asserted at zero. Writes only under P512Z/.
"""
import glob, json, pickle, re, sys
from pathlib import Path
import pyomo.environ as pe
import p512_k_no_solve_kkt_forensic as K

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/SRP1/Results/P512Z'
SPEC = json.load(open(OUT / 'frozen_formula_spec.json'))
K.install_solver_guards()

def sha(p): return K.sha(Path(p))
def sol_primals(pat):
    f = Path(glob.glob(pat)[0]); d = K.parse_sol_file(f)
    return d['primals'], str(f), d['n']
def logval(log, pat, last=True):
    hits = [l for l in open(log, errors='replace') if re.search(pat, l)]
    return hits[-1] if (hits and last) else (hits[0] if hits else None)
def final_obj(log):
    ls = open(log, errors='replace').read().splitlines()
    i = max(j for j, l in enumerate(ls) if l.startswith('Number of Iterations'))
    for l in ls[i:]:
        m = re.match(r'Objective\.+:\s+\S+\s+(\S+)', l)
        if m: return float(m.group(1))
def iters(log):
    return int(re.search(r':\s*(\d+)', logval(log, r'Number of Iterations')).group(1))
def nvars_log(log):
    return int(re.search(r':\s*(\d+)', logval(log, r'Total number of variables')).group(1))
def equal_bound_names(model_path, key=None):
    pay = pickle.load(open(model_path, 'rb'))
    m = pay['model'] if 'model' in pay else pay['model']
    return {v.name for v in m.component_data_objects(pe.Var, active=None)
            if v.lb is not None and v.ub is not None and v.lb == v.ub and not v.fixed}

FIX = {
 'W_TARGET_cycle20': dict(
   dirp='data/SRP1/Results/P512W', arms={'LOW':'LOW_1e-6','HIGH':'HIGH_1e-4'},
   logname='optim_log_case33_3_2025_Spring.log',
   smap='data/SRP1/Results/P512R/cycle20_prepared/original_mapping.json',
   model='data/SRP1/Results/P512R/cycle20_pre_setup/snapshot.pkl',
   baseline_kind='solverresults',
   baseline_pkl='data/SRP1/Results/P512R/cycle20_target_result.pkl',
   baseline_log='data/SRP1/Results/P512R/cycle20_target.log',
   nl_gate=['data/SRP1/Results/P512R/cycle20_prepared/original.nl',
            'data/SRP1/Results/P512W/LOW_1e-6/prepared.nl',
            'data/SRP1/Results/P512W/HIGH_1e-4/prepared.nl'],
   day='Spring'),
 'X_DSO_case33_2_cycle7': dict(
   dirp='data/SRP1/Results/P512X', arms={'LOW':'LOW_1e-6','HIGH':'HIGH_1e-4'},
   logname='optim_log_case33_2_2025_Autumn.log',
   smap='data/SRP1/Results/P512X/BASELINE/prepared_mapping.json',
   model='data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
   baseline_kind='sol', day='Autumn'),
 'Y_TSO_case9_cycle7': dict(
   dirp='data/SRP1/Results/P512Y', arms={'LOW':'LOW_1e-7','HIGH':'HIGH_1e-5'},
   logname='optim_log_case9_2025_Summer.log',
   smap='data/SRP1/Results/P512Y/BASELINE/prepared_mapping.json',
   model='data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
   baseline_kind='sol', day='Summer'),
}
W = {'Spring': 92, 'Summer': 91, 'Autumn': 91, 'Winter': 91}
FAMS = ['expected_interface_vmag', 'expected_interface_pf_p', 'expected_interface_pf_q']
TH = SPEC['threshold_application']

def region(x):
    return 'EQUIVALENT' if x <= 1e-6 else ('MATERIALLY DIFFERENT' if x > 1e-3 else 'INDETERMINATE')

res = {'spec_sha256': sha(OUT / 'frozen_formula_spec.json'), 'fixtures': {}, 'mismatch_log': []}
for fid, f in FIX.items():
    r = {'sources': {}}
    smap = json.load(open(f['smap'])); r['sources']['symbol_map'] = f['smap']
    idx2name = {int(s[1:]): n for s, n in smap['mapping'] if s.startswith('v')}
    name2idx = {n: i for i, n in idx2name.items()}
    eqb = equal_bound_names(f['model']); r['sources']['model_for_bounds'] = f['model']
    # baseline
    blog = f.get('baseline_log') or f"{f['dirp']}/BASELINE/logs/{f['logname']}"
    r['sources']['baseline_log'] = blog
    if f['baseline_kind'] == 'sol':
        b, bsrc, n_nl = sol_primals(f"{f['dirp']}/BASELINE/used_*.sol"); bB = None
    else:
        sr = pickle.load(open(f['baseline_pkl'], 'rb')); sv = sr.solution(0).variable
        n_nl = len(idx2name)
        b = [sv[f'v{i}']['Value'] if f'v{i}' in sv else 0.0 for i in range(n_nl)]        # method A
        bB = {n: (sv[s]['Value'] if s in sv else 0.0) for s, n in smap['mapping'] if s.startswith('v')}  # method B
        bsrc = f['baseline_pkl']
        r['nl_hash_gate'] = {'hashes': [sha(p) for p in f['nl_gate']],
                             'all_equal': len({sha(p) for p in f['nl_gate']}) == 1}
    r['sources']['baseline_primals'] = bsrc
    n_log = nvars_log(blog)
    excl = {name2idx[n] for n in eqb if n in name2idx}
    r['columns'] = {'n_nl': n_nl, 'n_opt': n_nl - len(excl), 'n_removed': len(excl),
                    'n_opt_log': n_log, 'cross_check_ok': (n_nl - len(excl)) == n_log}
    r['weight'] = {'N_year': 5, 'D_day': W[f['day']], 'annualization': 1.0,
                   'weight': 5 * W[f['day']]}
    bobj = final_obj(blog); biter = iters(blog)
    r['baseline'] = {'objective': bobj, 'iterations': biter}
    for lab, sub in f['arms'].items():
        alog = f"{f['dirp']}/{sub}/logs/{f['logname']}"
        v, vsrc, _ = sol_primals(f"{f['dirp']}/{sub}/used_*.sol")
        a = {'sources': {'primals': vsrc, 'log': alog}}
        for setname, cols in (('C_nl', range(n_nl)), ('C_opt', [i for i in range(n_nl) if i not in excl])):
            worst = 0.0; arg = None; ncols = 0
            for i in cols:
                d = abs(v[i] - b[i]) / max(1.0, abs(b[i]))
                if d > 1e-3: ncols += 1
                if d > worst: worst, arg = d, i
            a[setname] = {'max_scaled_primal': worst, 'argmax_col': arg,
                          'argmax_name': idx2name.get(arg), 'n_cols_gt_1e-3': ncols,
                          'region': region(worst)}
        if bB is not None:   # method B for W
            worstB = 0.0; argB = None
            for n, i in name2idx.items():
                if i in excl or n not in bB: continue
                d = abs(v[i] - bB[n]) / max(1.0, abs(bB[n]))
                if d > worstB: worstB, argB = d, n
            a['C_opt_methodB'] = {'max_scaled_primal': worstB, 'argmax_name': argB,
                                  'agrees_with_methodA': abs(worstB - a['C_opt']['max_scaled_primal']) <= 1e-12}
        iface = {}
        for fam in FAMS:
            cols = [i for n, i in name2idx.items() if n.startswith(fam + '[')]
            if not cols: continue
            w = max(abs(v[i] - b[i]) / max(1.0, abs(b[i])) for i in cols)
            iface[fam] = {'n_entries': len(cols), 'max_scaled': w, 'region': region(w)}
        a['interface'] = {'families_found': [x for x in FAMS if any(n.startswith(x + '[') for n in name2idx)],
                          'families_compared': sorted(iface), 'per_family': iface,
                          'worst_region': region(max(x['max_scaled'] for x in iface.values()))}
        aobj = final_obj(alog); aiter = iters(alog)
        ro = abs(aobj - bobj) / max(1.0, abs(bobj))
        a['objective'] = {'value': aobj, 'relative_difference': ro, 'region': region(ro)}
        a['weighted_obj_planning_units'] = abs(aobj - bobj) * r['weight']['weight']
        a['iterations'] = {'value': aiter, 'ratio_change': abs(aiter - biter) / biter,
                           'material': abs(aiter - biter) / biter > 0.5}
        a['branch'] = 'DIFFERENT' if (a['C_opt']['max_scaled_primal'] > 1e-3 and ro > 1e-3) else 'EQUIVALENT'
        r[lab] = a
    res['fixtures'][fid] = r

res['zero_solver'] = {'counters': dict(K._SOLVE_COUNTERS),
                      'zero_confirmed': all(x == 0 for x in K._SOLVE_COUNTERS.values())}
K.uninstall_solver_guards()
json.dump(res, open(OUT / 'classification.json', 'w'), indent=1, default=str)
print('[P512Z] done; zero_solver =', res['zero_solver']['counters'])
for fid, r in res['fixtures'].items():
    print(f"\n== {fid}  cols nl={r['columns']['n_nl']} opt={r['columns']['n_opt']} "
          f"removed={r['columns']['n_removed']} log={r['columns']['n_opt_log']} ok={r['columns']['cross_check_ok']} w={r['weight']['weight']}")
    if 'nl_hash_gate' in r: print('   NL hash gate all_equal =', r['nl_hash_gate']['all_equal'])
    for lab in ('LOW', 'HIGH'):
        a = r[lab]
        print(f"   {lab}: C_opt={a['C_opt']['max_scaled_primal']:.6e} ({a['C_opt']['argmax_name']}) "
              f"n>1e-3={a['C_opt']['n_cols_gt_1e-3']} | C_nl={a['C_nl']['max_scaled_primal']:.6e} "
              f"n>1e-3={a['C_nl']['n_cols_gt_1e-3']}")
        if 'C_opt_methodB' in a: print(f"      methodB={a['C_opt_methodB']['max_scaled_primal']:.6e} agrees={a['C_opt_methodB']['agrees_with_methodA']}")
        print(f"      obj rel={a['objective']['relative_difference']:.3e} [{a['objective']['region']}] "
              f"weighted={a['weighted_obj_planning_units']:.3e} pu | iters {a['iterations']['value']} ratio={a['iterations']['ratio_change']:.3f}")
        print(f"      interface compared={a['interface']['families_compared']}")
        for fam, d in a['interface']['per_family'].items(): print(f"        {fam}: {d['max_scaled']:.3e} [{d['region']}] n={d['n_entries']}")
