"""P5.12-W breadth replay: ONE fixture, ONE variant, ONE solve, fresh process.

Usage: python -B p512_w_breadth_replay.py <variant_label> <warm_start_bound_push>
Reloads the frozen cycle-20 target fixture, runs production solver setup once,
overrides exactly one IPOPT option, gates identity against the frozen prepared
capture, then solves once. No retry, no second solve.
"""
import json, pickle, shutil, sys, time
from pathlib import Path
import pyomo.environ as pe
from pyomo.opt.base.solvers import OptSolver
import network as N
import p512_r_presolve_recapture as H

ROOT = Path(__file__).resolve().parent
FIX = ROOT / 'data/SRP1/Results/P512R/cycle20_pre_setup'
PREP = ROOT / 'data/SRP1/Results/P512R/cycle20_prepared'
OUT_ROOT = ROOT / 'data/SRP1/Results/P512W'
PRE_DIGEST = '393ce242f1665ecb9d1738024476282c1c5d6f43876e0bf363858ad5f813be5c'
PREP_DIGEST = '58e2a063837cabdc75208ea0188882e88202f598cfdcb3746b08af665186c8e4'
PREP_NL = 'fbe9e6e846f6b5b51febdc26ab345315e150d0033979a818a8a909806817dbda'

label, value = sys.argv[1], float(sys.argv[2])
out = OUT_ROOT / label
assert not out.exists(), f'{out} exists; refusing to overwrite'
out.mkdir(parents=True)
(out / 'logs').mkdir()
rec = {'fixture_id': 'TARGET_cycle20', 'variant': label, 'warm_start_bound_push': value}

solves = {'n': 0}
_orig = OptSolver.solve
def counted(self, model, *a, **kw):
    solves['n'] += 1
    assert solves['n'] <= 1, 'more than one solve attempted'
    return _orig(self, model, *a, **kw)
OptSolver.solve = counted

payload = pickle.load(open(FIX / 'snapshot.pkl', 'rb'))
model, net, params = payload['model'], payload['network'], payload['params']
rec['pre_setup_digest_ok'] = (H.digest(H.model_state(model)) == PRE_DIGEST)
rec['boundary'] = payload['boundary']; rec['cycle'] = payload['cycle']
rec['target'] = payload['target']; rec['from_warm_start'] = payload['from_warm_start']
assert rec['pre_setup_digest_ok'] and payload['cycle'] == 20

net.logs_dir = str(out / 'logs')
solver, log_path, ctx = N._create_smopf_solver(net, model, params, from_warm_start=True)
solver.options['warm_start_bound_push'] = value

state_after = H.model_state(model)
rec['prepared_digest_ok'] = (H.digest(state_after) == PREP_DIGEST)
ex = H.export(model, solver, out, 'prepared')
rec['nl_sha256'] = ex['nl_sha256']; rec['nl_matches_frozen_prepared'] = (ex['nl_sha256'] == PREP_NL)
base_opts = json.load(open(PREP / 'manifest.json'))['effective_options']
now_opts = dict(solver.options)
diff = sorted({k for k in set(base_opts) | set(now_opts) if base_opts.get(k) != now_opts.get(k)})
rec['option_diff_keys'] = diff
rec['option_diff_ok'] = (diff == ['output_file', 'warm_start_bound_push'])
rec['effective_options'] = now_opts
assert rec['prepared_digest_ok'] and rec['nl_matches_frozen_prepared'] and rec['option_diff_ok'], rec

t0 = time.time()
res = solver.solve(model, tee=False, load_solutions=False, keepfiles=True)
rec['wall_seconds'] = time.time() - t0
rec['n_solves'] = solves['n']
rec['status'] = str(res.solver.status); rec['termination'] = str(res.solver.termination_condition)
for k in ('_problem_files', '_soln_file'):
    v = getattr(res, k, None) or getattr(solver, k, None)
    if v: rec[k] = str(v)
try:
    src = solver._problem_files[0]; shutil.copy(src, out / ('used_' + Path(src).name)); rec['used_nl'] = str(src)
except Exception as e:
    rec['used_nl_error'] = str(e)
try:
    sf = solver._soln_file; shutil.copy(sf, out / ('used_' + Path(sf).name)); rec['used_sol'] = str(sf)
except Exception as e:
    rec['used_sol_error'] = str(e)
pickle.dump(res, open(out / 'solver_results.pkl', 'wb'))
with open(out / 'variant_record.json', 'w') as f:
    json.dump(rec, f, indent=1, default=str)
OptSolver.solve = _orig
print(f"[P512W] {label} wsbp={value:g} -> {rec['termination']} solves={rec['n_solves']} gates(pre/prep/nl/opt)="
      f"{rec['pre_setup_digest_ok']}/{rec['prepared_digest_ok']}/{rec['nl_matches_frozen_prepared']}/{rec['option_diff_ok']}")
