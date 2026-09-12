"""P5.12-X comparator replay: ONE fixture, ONE variant, ONE solve, fresh process.

Usage: python -B p512_x_comparator_replay.py <variant_label> <warm_start_bound_push>

Reconstructs only the missing context (network/params) from fresh_planning; the
preserved pre-solve model is authoritative for all numerical state.
"""
import json, pickle, shutil, sys, time
from pathlib import Path
import pyomo.environ as pe
from pyomo.opt.base.solvers import OptSolver
import network as N
import p56a_oracle as O
import p512_r_presolve_recapture as H

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / 'data/SRP1/Results/P512Y'
FIX = ROOT / 'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl'
HIST_OPTS = {  # historical TSO cycle-7 option echo, excluding output_file
    'acceptable_iter': 0, 'acceptable_tol': 1e-05, 'bound_frac': 1e-06, 'bound_push': 1e-06,
    'compl_inf_tol': 0.0005, 'file_append': 'yes', 'file_print_level': 6,
    'linear_solver': 'ma97', 'slack_bound_frac': 1e-06, 'slack_bound_push': 1e-06,
    'tol': 1e-05, 'warm_start_bound_frac': 1e-06, 'warm_start_bound_push': 1e-06,
    'warm_start_init_point': 'yes', 'warm_start_mult_bound_push': 1e-06,
    'warm_start_slack_bound_frac': 1e-06, 'warm_start_slack_bound_push': 1e-06}

label, value = sys.argv[1], float(sys.argv[2])
out = OUT_ROOT / label
assert not out.exists(), f'{out} exists; refusing to overwrite'
out.mkdir(parents=True); (out / 'logs').mkdir()
rec = {'fixture_id': 'TSO_case9_2025_Summer_cycle7', 'variant': label,
       'warm_start_bound_push': value}

solves = {'n': 0}
_orig = OptSolver.solve
def counted(self, model, *a, **kw):
    solves['n'] += 1
    assert solves['n'] <= 1, 'more than one solve attempted'
    return _orig(self, model, *a, **kw)
OptSolver.solve = counted

payload = pickle.load(open(FIX, 'rb'))
md, model = payload['metadata'], payload['model']
rec['metadata'] = md
rec['preserved_model_digest'] = H.digest(H.model_state(model))

O.WORK_DIR = str(out / 'evals')
planning = O.fresh_planning('p512y')
net = planning.transmission_network.network[md['year']][md['day']]
params = planning.transmission_network.params
rec['identity_ok'] = (net.name == md['network_name'] and net.year == md['year']
                      and str(net.day) == str(md['day']) and net.is_transmission)
net.logs_dir = str(out / 'logs')

solver, log_path, ctx = N._create_smopf_solver(net, model, params,
                                               from_warm_start=md['from_warm_start'])
solver.options['warm_start_bound_push'] = value
now = {k: v for k, v in dict(solver.options).items() if k != 'output_file'}
expected = dict(HIST_OPTS)
if label != 'BASELINE':
    expected['warm_start_bound_push'] = value
diff = sorted({k for k in set(now) | set(expected)
               if str(now.get(k)) != str(expected.get(k))})
rec['option_diff_vs_historical'] = diff
rec['option_gate_ok'] = (diff == [])
rec['effective_options'] = dict(solver.options)
ex = H.export(model, solver, out, 'prepared')
rec['nl_sha256'] = ex['nl_sha256']; rec['mapping_sha256'] = ex['mapping_sha256']
assert rec['identity_ok'] and rec['option_gate_ok'], rec

t0 = time.time()
res = solver.solve(model, tee=False, load_solutions=False, keepfiles=True)
rec['wall_seconds'] = time.time() - t0; rec['n_solves'] = solves['n']
rec['status'] = str(res.solver.status); rec['termination'] = str(res.solver.termination_condition)
for attr, key in (('_problem_files', 'used_nl'), ('_soln_file', 'used_sol')):
    try:
        src = getattr(solver, attr)
        src = src[0] if isinstance(src, (list, tuple)) else src
        shutil.copy(src, out / ('used_' + Path(src).name)); rec[key] = str(src)
    except Exception as e:
        rec[key + '_error'] = str(e)
pickle.dump(res, open(out / 'solver_results.pkl', 'wb'))
json.dump(rec, open(out / 'variant_record.json', 'w'), indent=1, default=str)
OptSolver.solve = _orig
print(f"[P512X] {label} wsbp={value:g} -> {rec['termination']} solves={rec['n_solves']} "
      f"identity={rec['identity_ok']} optgate={rec['option_gate_ok']} nl={rec['nl_sha256'][:16]}")
