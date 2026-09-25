"""
P5.15 Addendum 48, Planner task W90 -- option (b) measured at zero solves on the 3 x 3 instance; frozen stage spec v35
(predecessor v34 e1940b92, NOT edited); the TWO-ARM 3 x 3 smoke gate ((b) on vs (b) off) and the 3 x 3 pair under (b).
BUILT AND FROZEN IN W90; THE SMOKE AND THE PAIR ARE NOT RUN IN W90 (the author reboots the Mac first).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 48 (tight tail adopted; 3 x 3 pair on the Mac, now; option (b)
`release_solution_bookkeeping` on, measured first at zero solves at 3 x 3; persistence only if the measured runtime peak
with persistence <= 0.85 x memory available after the reboot; without persistence the hull polish is omitted at 3 x 3;
the smoke becomes a two-arm 3-cycle bitwise comparison that measures the 3 x 3 runtime peak directly; G6's final scope
written into the spec, the two earlier re-scopings recorded as post-hoc; `identity_holds` recomputed or dropped;
predictions restated before launch; frozen eval keys, prefix draw and the SRP1 (b) gate carry over); Planner task W90
(including the Planner's ruling on W89 question 4: a tier-2 final accepted attempt in T must not fail G6 through
non-vacuity).

WHAT OPTION (b) IS (read from the code). `SolverParameters.release_solution_bookkeeping` (P5.15 Addendum 29, W32):
after a successful network solve and `model.solutions.load_from(result)`, `network._run_smopf` calls
`network._release_solution_bookkeeping(model, result)`, which clears `model.solutions` (the ModelSolution: one
(component, entry) pair per variable and per active constraint) and `result.solution` (the SolverResults' own entry
dicts). Var values, the dual / ipopt_z*_out / ipopt_z*_in suffixes and `result.solver` are untouched. The committed
SRP1 (b) bitwise gate (P515S49/memory_fix_gate) reproduced the committed C* trajectory bitwise over two cycles with it
on. The campaign harness had no path for it before W90; W90 adds one (`p515_s44_campaign_harness`: an entry option
`release_solution_bookkeeping`, applied in the child's config hook through the SRP1 gate's own setter, read back, with a
pass-through call counter; it NEVER enters `evaluation_key`, so the v34 keys carry over).

ITEM 1 -- THE ZERO-SOLVE MEASUREMENT (`--b-probe`). The bookkeeping only exists after a solve, so building the models
with the switch on and off measures nothing. The probe REPRODUCES THE OBJECTS WITHOUT SOLVING, through Pyomo's and
production's own code: for every network block of production's ADMM model build (the W89 memory probe's build, unit
candidate), (1) `Block.write(format='nl', symbolic_solver_labels=False)` -- the NL writer's legacy call path that
`SolverFactory('ipopt').solve` uses (it registers the symbol map in `model.solutions`); (2) a SYNTHETIC IPOPT .sol file
with the real file's layout (message, Options 3 1 1 0, m, m, n, n, m duals, n primal values, `objno 0 0`, then the
`ipopt_zU_out` and `ipopt_zL_out` float variable suffixes) holding one zL for every variable with a finite lower bound
(NL bound types 0, 2, 4) and one zU for every variable with a finite upper bound and unequal bounds (types 0, 1) --
IPOPT writes only non-zero multipliers, and an interior-point multiplier on a finite bound is non-zero; (3) Pyomo's own
reader (`ReaderFactory(ResultsFormat.sol)`) with the model's active IMPORT suffixes -- what `SystemCallSolver
.process_output` calls; (4) `OptSolver.solve`'s `load_solutions=False` tail (result._smap = the symbol map; the map
deleted from the model); (5) production's `model.solutions.load_from(result)` (network._run_smopf's call); (6)
production's `helper_functions.replace_warm_start_suffix` for zL / zU (the next warm solve's copy, so every block holds
the steady-state suffix set); and, in the ON arm only, (7) production's `network._release_solution_bookkeeping(model,
result)` right after the load -- exactly where `_run_smopf` calls it. The OFF arm keeps every SolverResults in a dict, as
production keeps `results[year][day]`. Two FRESH processes (arms `s53_3x3_off`, `s53_3x3_on`), same build, same blocks;
the (b) saving is the difference of the process's memory after all 80 loads (RSS and macOS phys_footprint), each net of
its own post-build level. The ON arm then pickles the models through the child's persist callable
(`p515_s42_exact_fix_rerun._persist_certified_models`) into scratch with a 20 ms peak sampler -- the persistence transient
with realistic suffix data and no bookkeeping, i.e. the (b)-on state. `srp1_off` validates the method: the synthetic
.sol's entry structure against a REAL IPOPT .nl/.sol pair on disk, and the per-block object sizes (the committed
`p515_s49_memory_profile.attribution` walk, by import) against the committed W32 attribution; it also checks
`set_release_solution_bookkeeping` on a planning object (True / False read back). ZERO SOLVES: nothing calls
`OptSolver.solve` or `SystemCallSolver._execute_command`; every armed permitted=() guard verify(0) == [].
LIMITATIONS, stated in the artifact: values are synthetic (structure, not numbers, is reproduced); IPOPT's own process
and workspace, the pristine snapshot clones, AA memory and the terminal capture are not reproduced (they are (b)-
independent); RSS / footprint freed by a release is allocator-dependent; one measurement per arm (no repeats); the
pickle transient was noisy in W89 (+5.27 committed, +6.56 in an earlier uncommitted run).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --b-probe {s53_3x3_off,s53_3x3_on,srp1_off} --scratch D   ZERO SOLVES. -> <root>/memory_probe_b/b_probe_<W>.json
  --freeze-spec                     ZERO SOLVES. Frozen stage spec v35 (write-once, named by its sha256; predecessor v34).
  --stage smoke --freeze            ZERO SOLVES. The TWO smoke campaign specs (arms bon / boff; distinct campaign ids,
                                    so distinct working dirs), pinning v35.
  --stage smoke --run --spec-sha256-bon A --spec-sha256-boff B   NOT RUN IN W90. After the reboot: the (b)-on arm, then
                                    the (b)-off arm (x0, cap 3, each through H.evaluate), sequential, alone; the smoke
                                    gate S1-S17 and the persistence margin rule, recorded in <root>/smoke_gate/.
  --stage pair --freeze             NOT RUN IN W90. Requires the smoke gate committed, clean and PASS; post-certification
                                    as the smoke gate's margin-rule verdict.
  --stage pair --run --spec-sha256 S    NOT RUN IN W90. The two cells at concurrency 1 (sequential); gates G1-G12.

EXACT COMMANDS (repo root):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --b-probe <W> --scratch <dir outside the repo> > data/SRP1/Results/P515S53/w90_3x3/b_probe_<W>_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --freeze-spec > data/SRP1/Results/P515S53/w90_3x3/freeze_spec_v35_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --stage smoke --freeze > data/SRP1/Results/P515S53/w90_3x3/smoke_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --stage smoke --run --spec-sha256-bon <A> --spec-sha256-boff <B> \\
      > data/SRP1/Results/P515S53/w90_3x3/smoke_run_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --stage pair --freeze > data/SRP1/Results/P515S53/w90_3x3/pair_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w90_3x3_campaign.py \\
      --stage pair --run --spec-sha256 <S> > data/SRP1/Results/P515S53/w90_3x3/pair_run_launch.log 2>&1
Exit codes: probes / freezes 0 done, 1 precondition / guard failure; smoke --run 0 PASS / 1 FAIL; pair --run 0 every
gate holds and both cells certified, 2 a cell not certified (harness clean), 1 a gate / harness / guard / precondition
failure.
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W90 3x3 launcher (never solves)').install()

# The W89 3 x 3 launcher: its instance, memory model, cell formulas and checks, BY IMPORT (it arms its own permitted=()
# guard and, through its imports, the alpha-row, W86, W87, W88 and W89-step-1 guards -- all verified at 0 here).
import p515_s53_w89_3x3_campaign as W9  # noqa: E402
H = W9.H
X = W9.X     # the v32 G6 evaluator (W89 step 1)
L = W9.L     # the W86 launcher
A = W9.A     # the alpha-row launcher

GUARDS_LIFO = tuple(W9.GUARDS_LIFO) + (PARENT_GUARD,)
GUARD_NAMES = ('alpha_row_launcher', 'w86_launcher', 'w87', 'w88', 'w89_step1', 'w89_3x3_launcher', 'w90_3x3_parent')

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s53_w90_3x3_campaign'
STAGE_TEXT = ('P5.15 Addendum 48, W90 -- option (b) measured at zero solves on the 3 x 3 instance; stage spec v35; the '
              'two-arm 3 x 3 smoke ((b) on vs off, bitwise, measuring the runtime peak directly) and the 3 x 3 pair '
              '(x = 0 and the smallest node-7 unit, concurrency 1, (b) on, persistence by the margin rule)')
_P53 = W9._P53
ROOT_REL = os.path.join(_P53, 'w90_3x3')
SPEC_V34 = {'path': os.path.join(_P53, 'frozen_s53_spec_v34_e1940b92.json'),
            'sha256': 'e1940b927759d70d5fddcb95edaa001c4ac13072e2f0f6f7629ad4592878bdb7'}
SPEC_PREFIX = 'frozen_s53_spec_v35_'
SPEC_VERSION = 35
GIB = 1 << 30
MIB = 1 << 20

# ---- the SRP1 (b) bitwise gate that carries over (Addendum 48) -------------------------------------------------------
SRP1_B_GATE = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S49', 'memory_fix_gate', 'gate.json'),
               'sha256': 'acea13347c6ececc0bf62c676aed7506ddbe1872238c2e84815ef18eb064567b',
               'manifest': os.path.join('data', 'SRP1', 'Results', 'P515S49', 'memory_fix_gate', 'manifest_sha256.json')}
# ---- the committed W32 memory attribution (Addenda 29 / 32-34) -------------------------------------------------------
W32_PROFILE = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'memory_profile', 'memory_profile.json')
W32_PROFILE_SHA256 = 'c6540fa00fafd07aa7aab1118f9c4089f681b91c2d9a15c2c65b05bd4b123313'
# ---- a real IPOPT .nl / .sol pair on disk (UNTRACKED; hash-recorded when used) ---------------------------------------
REAL_PAIR = {'nl': os.path.join('data', 'SRP1', 'Results', 'P512ArmA', 'used_tmp27ntrqce.pyomo.nl'),
             'sol': os.path.join('data', 'SRP1', 'Results', 'P512ArmA', 'used_tmp27ntrqce.pyomo.sol')}

B_PROBES = ('s53_3x3_off', 's53_3x3_on', 'srp1_off')
B_PROBE_DIR_REL = os.path.join(ROOT_REL, 'memory_probe_b')
SYNTH_MESSAGE = 'Ipopt 3.14.18: Optimal Solution Found.'
PICKLE_MIN_AVAILABLE_GIB = 9.0      # the ON arm's pickle step is skipped (recorded) below this availability
PEAK_SAMPLER_S = 0.02

# ---- stages ----------------------------------------------------------------------------------------------------------
SMOKE_CAP = 3
STAGES = {
    'smoke_bon': {'campaign_id': 's53_w90_3x3_smoke_bon', 'labels': ('x0',), 'cap': SMOKE_CAP,
                  'release_solution_bookkeeping': True},
    'smoke_boff': {'campaign_id': 's53_w90_3x3_smoke_boff', 'labels': ('x0',), 'cap': SMOKE_CAP,
                   'release_solution_bookkeeping': False},
    'pair': {'campaign_id': 's53_w90_3x3_pair', 'labels': ('x0', 'n7_4h_e1'), 'cap': 500,
             'release_solution_bookkeeping': True},
}
SMOKE_ARMS = ('smoke_bon', 'smoke_boff')   # run order
CONCURRENCY = 1
SMOKE_GATE_DIR_REL = os.path.join(ROOT_REL, 'smoke_gate')
SMOKE_GATE_FILE = 'smoke_gate.json'
SMOKE_MANIFEST_FILE = 'smoke_manifest_sha256.json'
PAIR_RESULTS_FILE = 'campaign_results.json'
PAIR_MANIFEST_FILE = 'campaign_manifest_sha256.json'
MARGIN = 0.85
POST_CERT_NO_PERSIST = {'persist_certified_models': False, 'hull_polish': False}
POST_CERT_PERSIST = {'persist_certified_models': True, 'hull_polish': False}
EXTRA_CLEAN_FILES = tuple(W9.EXTRA_CLEAN_FILES) + (SCRIPT_NAME, 'p515_s53_w89_3x3_campaign.py', 'network.py',
                                                   'helper_functions.py', 'solver_parameters.py',
                                                   'p515_s49_memory_profile.py', 'p515_s42_exact_fix_rerun.py')
# per-cycle fields that measure time / memory, excluded from the bitwise comparison (everything else is compared)
NON_DETERMINISTIC_CYCLE_FIELDS = ('cycle_wall_s', 'rss_bytes', 'ru_maxrss_bytes', 'response_capture_s')
# per-solve record fields that are paths / time, excluded from the (reported) record comparison
NON_DETERMINISTIC_RECORD_KEYS = ('log_path', 'log', 'wall_s', 'wall', 'time', 'utc', 't_s', 'log_offset', 'path',
                                 'ipopt_log_path', 'solver_log_path', 'cpu_s', 'elapsed_s')


# ======================================================================================================================
#  utilities
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_state(rel):
    return L._git_state(rel)


def _committed_clean(rel):
    st = _git_state(rel)
    return bool(st.get('git_tracked') and st.get('git_clean'))


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in zip(GUARD_NAMES, GUARDS_LIFO)}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    """EVERY exit path: verify every guard at exactly 0, uninstall them LIFO, exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W90] guards {g} {extra_msg}')
    for guard in reversed(GUARDS_LIFO):
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    """Other live processes running THIS script (never this process or its ancestors); no pattern on a command line."""
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and (OWN_PROCESS_SUBSTRING in parts[1]
                                                             or W9.OWN_PROCESS_SUBSTRING in parts[1]):
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


def _mem_sample():
    import p515_s44_scale_measurement as S44
    import psutil
    return {'rss_bytes': psutil.Process().memory_info().rss, 'footprint_bytes': S44.phys_footprint(os.getpid()),
            'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class _PeakSampler:
    """Peak RSS and phys_footprint of THIS process while a step runs (a 20 ms sampler thread)."""

    def __init__(self):
        self.peak_rss = 0
        self.peak_footprint = 0
        self.n = 0
        self._halt = threading.Event()
        self._thread = None

    def _run(self):
        import p515_s44_scale_measurement as S44
        import psutil
        me = psutil.Process()
        while not self._halt.is_set():
            try:
                self.peak_rss = max(self.peak_rss, me.memory_info().rss)
                self.peak_footprint = max(self.peak_footprint, S44.phys_footprint(os.getpid()) or 0)
            except Exception:  # noqa: BLE001 -- a missed sample is not an error
                pass
            self.n += 1
            self._halt.wait(PEAK_SAMPLER_S)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._thread.join(timeout=2)
        return False


# ======================================================================================================================
#  ITEM 1 -- the zero-solve (b) probe
# ======================================================================================================================
def nl_header_and_bound_types(path):
    """(n_vars, n_cons, bound type per variable) from an .nl file: header line 2, and the `b` segment (one line per
    variable, first token the AMPL bound type 0 lb<=x<=ub, 1 x<=ub, 2 x>=lb, 3 free, 4 x==c, 5 complementarity)."""
    with open(path) as handle:
        handle.readline()
        head = handle.readline().split()
        n, m = int(head[0]), int(head[1])
        types = None
        for line in handle:
            tok = line.split()
            if tok and tok[0] == 'b':
                types = [int(handle.readline().split()[0]) for _ in range(n)]
                break
    if types is None:
        types = [3] * n    # an .nl with no `b` segment: every variable free
    return n, m, types


def suffix_indices(types):
    """IPOPT writes only non-zero bound multipliers: zL for every finite lower bound (types 0, 2, 4), zU for every
    finite upper bound with unequal bounds (types 0, 1). For an equal-bound variable (type 4) IPOPT reports exactly ONE
    of the two (the sign of its reduced cost) -- validated on a real pair (`real_pair_validation`)."""
    zl = [i for i, t in enumerate(types) if t in (0, 2, 4)]
    zu = [i for i, t in enumerate(types) if t in (0, 1)]
    return zl, zu


def write_synthetic_sol(path, n, m, types):
    zl, zu = suffix_indices(types)
    with open(path, 'x') as f:
        f.write(f'{SYNTH_MESSAGE}\n\nOptions\n3\n1\n1\n0\n{m}\n{m}\n{n}\n{n}\n')
        f.write('1e-06\n' * m)
        f.write('0.5\n' * n)
        f.write('objno 0 0\n')
        f.write(f'suffix 4 {len(zu)} 13 0 0\nipopt_zU_out\n')
        f.writelines(f'{i} -1e-06\n' for i in zu)
        f.write(f'suffix 4 {len(zl)} 13 0 0\nipopt_zL_out\n')
        f.writelines(f'{i} 1e-06\n' for i in zl)
    return {'n_zL': len(zl), 'n_zU': len(zu)}


def read_sol(path, suffixes):
    import pyomo.environ  # noqa: F401 -- registers the plugins (the 'sol' results reader among them)
    from pyomo.opt.base.results import ReaderFactory
    from pyomo.opt.base.formats import ResultsFormat
    reader = ReaderFactory(ResultsFormat.sol)
    return reader(path, suffixes=list(suffixes))


def _solution_entry_counts(result):
    sol = result.solution(0)
    return {k: len(getattr(sol, k)) for k in ('variable', 'constraint', 'objective', 'problem')}


def _suffix_entry_totals(result):
    sol = result.solution(0)
    out = Counter()
    for entry in sol.variable.values():
        for k in entry:
            out[f'variable:{k}'] += 1
    for entry in sol.constraint.values():
        for k in entry:
            out[f'constraint:{k}'] += 1
    return dict(sorted(out.items()))


def synthetic_load(model, stem, nl_dir, release):
    """Steps (1)-(7) of the module docstring on ONE block. Returns (result, per-block record)."""
    import helper_functions as HF
    import network as NET
    from pyomo.core.base.suffix import active_import_suffix_generator
    t0 = time.time()
    nl_path = os.path.join(nl_dir, f'{stem}.nl')
    sol_path = os.path.join(nl_dir, f'{stem}.sol')
    _fname, smap_id = model.write(nl_path, format='nl', io_options={'symbolic_solver_labels': False})
    n, m, types = nl_header_and_bound_types(nl_path)
    nl_bytes = os.path.getsize(nl_path)
    zinfo = write_synthetic_sol(sol_path, n, m, types)
    sol_bytes = os.path.getsize(sol_path)
    suffixes = [name for name, _comp in active_import_suffix_generator(model)]
    result = read_sol(sol_path, suffixes)
    os.remove(nl_path)
    os.remove(sol_path)
    # OptSolver.solve, load_solutions=False: the symbol map moves from the model to the result
    result._smap_id = smap_id
    result._smap = model.solutions.symbol_map[smap_id]
    model.solutions.delete_symbol_map(smap_id)
    ok = HF.solver_result_succeeded(result)
    model.solutions.load_from(result)                       # network._run_smopf's call
    ms = model.solutions.solutions
    ms_entries = {k: len(v) for k, v in ms[0]._entry.items()} if ms else None
    res_entries = _solution_entry_counts(result)
    HF.replace_warm_start_suffix(model.ipopt_zL_in, model.ipopt_zL_out)    # the next warm solve's copy
    HF.replace_warm_start_suffix(model.ipopt_zU_in, model.ipopt_zU_out)
    if release:
        NET._release_solution_bookkeeping(model, result)     # where _run_smopf calls it, right after the load
    rec = {'n_nl_vars': n, 'n_nl_cons': m, 'bound_types': dict(sorted(Counter(types).items())), **zinfo,
           'nl_bytes': nl_bytes, 'sol_bytes': sol_bytes, 'import_suffixes': suffixes,
           'solver_result_succeeded': ok, 'model_solution_entries_before_release': ms_entries,
           'result_solution_entries_before_release': res_entries,
           'suffix_lengths': {s: len(getattr(model, s)) for s in ('dual', 'ipopt_zL_out', 'ipopt_zU_out',
                                                                   'ipopt_zL_in', 'ipopt_zU_in') if hasattr(model, s)},
           'released': bool(release),
           'after_release': ({'model_solutions_n': len(model.solutions.solutions),
                              'result_solution_n': len(result.solution)} if release else None),
           'wall_s': round(time.time() - t0, 3)}
    return result, rec


def real_pair_validation():
    """The synthetic .sol's ENTRY STRUCTURE against a real IPOPT .nl / .sol pair on disk (untracked files, hash-recorded):
    same n / m, same variable / constraint entry counts, same TOTAL of zL + zU entries (the split differs only on
    equal-bound variables, which IPOPT reports on one side by the sign of the reduced cost), same object bytes of the
    SolverResults' solution container (the committed W32 sizer)."""
    import p515_s49_memory_profile as MP
    from pyomo.core.base.component import ComponentBase
    if not all(os.path.isfile(_abs(p)) for p in REAL_PAIR.values()):
        return {'status': 'skipped', 'why': f'real pair not on disk: {REAL_PAIR}'}
    suffixes = ['dual', 'ipopt_zL_out', 'ipopt_zU_out']
    real = read_sol(_abs(REAL_PAIR['sol']), suffixes)
    n, m, types = nl_header_and_bound_types(_abs(REAL_PAIR['nl']))
    with tempfile.TemporaryDirectory(prefix='w90_realpair_') as d:
        p = os.path.join(d, 'synthetic.sol')
        zinfo = write_synthetic_sol(p, n, m, types)
        synth = read_sol(p, suffixes)
    r_tot, s_tot = _suffix_entry_totals(real), _suffix_entry_totals(synth)
    size_real = MP._sizer([real.solution], set(), (ComponentBase,))
    size_synth = MP._sizer([synth.solution], set(), (ComponentBase,))
    rs, ss = real.solution(0), synth.solution(0)
    entries_real = MP._sizer([rs.variable, rs.constraint], set(), (ComponentBase,))
    entries_synth = MP._sizer([ss.variable, ss.constraint], set(), (ComponentBase,))
    zl_real, zu_real = r_tot.get('variable:ipopt_zL_out', 0), r_tot.get('variable:ipopt_zU_out', 0)
    checks = {
        'n_m_equal': (len(real.solution(0).variable), len(real.solution(0).constraint)) == (n, m),
        'variable_entries_equal': len(real.solution(0).variable) == len(synth.solution(0).variable),
        'constraint_entries_equal': len(real.solution(0).constraint) == len(synth.solution(0).constraint),
        'dual_entries_equal': r_tot.get('constraint:Dual') == s_tot.get('constraint:Dual'),
        'zl_plus_zu_equal': zl_real + zu_real == zinfo['n_zL'] + zinfo['n_zU'],
        'entry_containers_bytes_equal': entries_real == entries_synth,
        'n_objects_equal': size_real['n_objects'] == size_synth['n_objects'],
        'whole_container_differs_only_by_message_strings': abs(size_real['container_bytes']
                                                               - size_synth['container_bytes']) <= 1024,
    }
    return {'status': 'evaluated', 'files': {k: {'path': v, 'sha256': _sha(v), 'git_tracked': _git_state(v)[
        'git_tracked']} for k, v in REAL_PAIR.items()},
            'real_message': str(real.solver.message), 'n': n, 'm': m,
            'bound_types': dict(sorted(Counter(types).items())),
            'real_suffix_entries': r_tot, 'synthetic_suffix_entries': s_tot,
            'real_zL_zU': [zl_real, zu_real], 'synthetic_zL_zU': [zinfo['n_zL'], zinfo['n_zU']],
            'size_real_solution_container': size_real, 'size_synthetic_solution_container': size_synth,
            'size_real_variable_constraint_entries': entries_real,
            'size_synthetic_variable_constraint_entries': entries_synth,
            'whole_container_bytes_difference': size_real['container_bytes'] - size_synth['container_bytes'],
            'whole_container_difference_note': ('the solution status_description / message strings: the real file is '
                                                'a max_iter exit, the synthetic an optimal one'),
            'checks': checks, 'all_checks_pass': all(checks.values())}


def _blocks(planning, tso_model, dso_models):
    tn = planning.transmission_network
    out = [('TSO', None, tn.name, y, d, tso_model[y][d]) for y in tn.years for d in tn.days]
    for node in sorted(dso_models):
        dn = planning.distribution_networks[node]
        out += [('DSO', node, dn.name, y, d, dso_models[node][y][d]) for y in dn.years for d in dn.days]
    return out


def _w32_committed_srp1_attribution():
    """The committed W32 attribution (SRP1 and paper, node 7, first year / first day, first solve) -- the figures the
    probe's per-entry sizes are compared with."""
    d = _load(W32_PROFILE)
    out = {}
    for inst in ('srp1', 'paper'):
        a = d['attribution'][inst]
        s = a['per_solve'][0]['attribution']
        ms = s['size_union_model_solutions_then_result']
        entries = s['model_solutions']['entries']['variable'] + s['model_solutions']['entries']['constraint']
        freeable = ms['model_solutions']['container_bytes'] + ms['result_solution_additional']['container_bytes']
        out[inst] = {'block_sizes': a['block_sizes'], 'model_solution_entries': s['model_solutions']['entries'],
                     'result_entries': s['result']['entries'], 'suffix_lengths': s['suffix_lengths'],
                     'union_container_bytes': freeable, 'union_leaf_bytes': ms['model_solutions']['leaf_bytes'],
                     'entries_total': entries, 'container_bytes_per_entry': freeable / entries}
    inc = d['increments']
    out['paper_release_rss_mib'] = {v: inc[f'paper|single|{v}']['solve1']['rss_self_action']['stats']
                                    for v in ('c', 'e')}
    out['srp1_release_rss_mib'] = {v: inc[f'srp1|single|{v}']['solve1']['rss_self_action']['stats']
                                   for v in ('c', 'e')}
    out['source'] = {'path': W32_PROFILE, 'sha256': _sha(W32_PROFILE), **_git_state(W32_PROFILE)}
    return out


def b_probe(which, started, scratch):
    import p515_s44_scale_measurement as S
    import p56a_oracle as O
    import shared_resources_planning as srp
    import p515_s42_exact_fix_rerun as EF
    import p515_s49_memory_profile as MP
    instance, arm = which.rsplit('_', 1)
    release = arm == 'on'
    tag = f'W90-BPROBE-{which}'
    out_rel = os.path.join(B_PROBE_DIR_REL, f'b_probe_{which}.json')
    if os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] {out_rel} exists (write-once)')
        _finish(1)
    others = _own_process_alive()
    if others:
        _log(f'[{tag} PRECONDITION FAILED] another copy of a 3 x 3 launcher is alive: {others}')
        _finish(1)
    marks = []
    t0 = time.time()

    def mark(stage):
        s = _mem_sample()
        marks.append({'stage': stage, 't_s': round(time.time() - t0, 3), **s})
        _log(f"[{tag}] {stage}: rss {s['rss_bytes'] / GIB:.3f} GiB, footprint {(s['footprint_bytes'] or 0) / GIB:.3f} "
             f"GiB, ru_maxrss {s['ru_maxrss_bytes'] / GIB:.3f} GiB")
        return marks[-1]

    mark('start (launcher, harness and production modules imported)')
    mem_start = L.memory_preflight(1)
    probe_instance = 'srp1' if instance == 'srp1' else 's53_3x3'
    case_rel, case_sha, derived = W9._probe_case(probe_instance, scratch)
    planning, read_dir, _wall = W9._read(case_rel, scratch, f'bprobe_{which}')
    mark('planning read (production reader)')
    setter_check = None
    if instance == 'srp1':   # the harness's setter, on a real planning object: True then False, read back
        setter_check = {'true': S.set_release_solution_bookkeeping(planning, True),
                        'false': S.set_release_solution_bookkeeping(planning, False)}
    sed = planning.shared_ess_data
    x = {(n, y): {'s': 0.0, 'e': 0.0} for n in sed.active_distribution_network_nodes for y in sed.years}
    ykey = next(y for y in sed.years if int(y) == W9.YEAR)
    x[(7, ykey)] = {'s': 0.25, 'e': 1.0}
    candidate = O.vector_to_candidate(planning, x)
    if derived:
        planning.params.admm.interface_deviation_premium = dict(W9.PREMIUM)
    premium = planning.params.admm.interface_deviation_premium
    interceptor = S.Interceptor()
    tn = planning.transmission_network
    tn.optimize = interceptor.network(tn, 'tso')
    for dn in planning.distribution_networks.values():
        dn.optimize = interceptor.network(dn, 'dso')
    sed.optimize = interceptor.esso(sed)
    t_build = time.time()
    try:   # the W89 memory probe's build, verbatim
        cv, _dv = srp.create_admm_variables(planning)
        dso_models, _r1 = srp.create_distribution_networks_models(
            planning.distribution_networks, cv, candidate['total_capacity'], parallel_execution=False,
            premium_alpha=premium['alpha'], premium_floor=premium['floor'])
        tso_model, _r2 = srp.create_transmission_network_model(planning, cv, candidate['total_capacity'])
        esso_model, _r3 = srp.create_shared_energy_storage_model(sed, cv, candidate['investment'])
        srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
        srp._prepare_transmission_objectives_for_admm(tn, tso_model)
    finally:
        del tn.optimize
        for dn in planning.distribution_networks.values():
            del dn.optimize
        del sed.optimize
    build_wall = time.time() - t_build
    m_built = mark('ADMM models built (the W89 probe build; network blocks + 3 ESSO)')
    blocks = _blocks(planning, tso_model, dso_models)
    nl_dir = tempfile.mkdtemp(prefix=f'w90_nl_{which}_', dir=scratch)
    results, per_block = {}, []
    t_loop = time.time()
    for i, (agent, node, name, y, d, model) in enumerate(blocks):
        stem = f'{i:03d}_{agent}_{node}_{y}_{d}'
        result, rec = synthetic_load(model, stem, nl_dir, release)
        results[(agent, node, y, d)] = result       # kept, as production keeps results[year][day]
        s = _mem_sample()
        per_block.append({'i': i, 'agent': agent, 'node': node, 'network': name, 'year': str(y), 'day': str(d),
                          **rec, 'rss_after_bytes': s['rss_bytes'], 'footprint_after_bytes': s['footprint_bytes']})
        if i % 10 == 0 or i == len(blocks) - 1:
            _log(f"[{tag}] block {i + 1}/{len(blocks)} {agent} {name} {y} {d}: n {rec['n_nl_vars']} m "
                 f"{rec['n_nl_cons']} zL {rec['n_zL']} zU {rec['n_zU']} released {rec['released']} rss "
                 f"{s['rss_bytes'] / GIB:.3f} GiB footprint {(s['footprint_bytes'] or 0) / GIB:.3f} GiB "
                 f"({rec['wall_s']} s)")
    loop_wall = time.time() - t_loop
    m_all = mark(f"all {len(blocks)} network blocks loaded ({'bookkeeping RELEASED after each load' if release else 'bookkeeping KEPT'})")
    gc.collect()
    m_gc = mark('after gc.collect()')
    attribution = None
    if not release:   # the committed W32 object walk, per block (allocates: after the memory marks)
        t_a = time.time()
        attribution = []
        for (agent, node, name, y, d, model) in blocks:
            a = MP.attribution(model, results[(agent, node, y, d)])
            u = a['size_union_model_solutions_then_result']
            freeable = u['model_solutions']['container_bytes'] + (u['result_solution_additional'] or {}).get(
                'container_bytes', 0)
            ent = a['model_solutions']['entries'] or {}
            attribution.append({'agent': agent, 'node': node, 'network': name, 'year': str(y), 'day': str(d),
                                'attribution': a, 'freeable_container_bytes': freeable,
                                'entries_total': (ent.get('variable') or 0) + (ent.get('constraint') or 0)})
        attribution = {'per_block': attribution, 'wall_s': time.time() - t_a,
                       'total_freeable_container_bytes': sum(b['freeable_container_bytes'] for b in attribution),
                       'total_leaf_bytes_union': sum(b['attribution']['size_union_model_solutions_then_result'][
                           'model_solutions']['leaf_bytes'] for b in attribution),
                       'total_entries': sum(b['entries_total'] for b in attribution)}
        mark('after the W32 attribution walk (reported only)')
    pickle_step = None
    if release and instance != 'srp1':
        m_pre = L.memory_preflight(1)
        avail = m_pre.get('available_bytes') or 0
        if avail < PICKLE_MIN_AVAILABLE_GIB * GIB:
            pickle_step = {'status': 'skipped', 'why': f'available {avail / GIB:.2f} GiB < {PICKLE_MIN_AVAILABLE_GIB}',
                           'memory_preflight': m_pre}
        else:
            persist_dir = tempfile.mkdtemp(prefix=f'w90_persist_{which}_', dir=scratch)
            before = _mem_sample()
            tp = time.time()
            with _PeakSampler() as sampler:
                persisted = EF._persist_certified_models({'tso': tso_model, 'dso': dso_models, 'esso': esso_model},
                                                         persist_dir)
            after = _mem_sample()
            pkl = _abs(persisted['path']) if not os.path.isabs(persisted['path']) else persisted['path']
            size = os.path.getsize(pkl)
            os.remove(pkl)
            pickle_step = {
                'status': 'measured', 'callable': 'p515_s42_exact_fix_rerun._persist_certified_models (tso + dso)',
                'size_bytes': size, 'wall_s': time.time() - tp, 'before': before, 'after': after,
                'memory_preflight_before': m_pre,
                'sampler': {'peak_rss_bytes': sampler.peak_rss, 'peak_footprint_bytes': sampler.peak_footprint,
                            'n_samples': sampler.n, 'period_s': PEAK_SAMPLER_S},
                'transient_rss_sampled_bytes': sampler.peak_rss - before['rss_bytes'],
                'transient_footprint_sampled_bytes': sampler.peak_footprint - (before['footprint_bytes'] or 0),
                'transient_ru_maxrss_bytes_W89_definition': after['ru_maxrss_bytes'] - before['rss_bytes'],
                'ru_maxrss_before_exceeds_rss_before': before['ru_maxrss_bytes'] > before['rss_bytes'],
                'retained_rss_bytes': after['rss_bytes'] - before['rss_bytes'],
                'file': 'written to scratch, measured, DELETED (never in the repository)'}
            mark('certified-model pickle written, measured and deleted')
    validation = real_pair_validation() if instance == 'srp1' else None
    committed_w32 = _w32_committed_srp1_attribution() if instance == 'srp1' else None
    n_blocks = len(blocks)
    totals = {
        'n_blocks': n_blocks, 'n_tso_blocks': sum(1 for b in per_block if b['agent'] == 'TSO'),
        'n_dso_blocks': sum(1 for b in per_block if b['agent'] == 'DSO'),
        'sum_nl_vars': sum(b['n_nl_vars'] for b in per_block), 'sum_nl_cons': sum(b['n_nl_cons'] for b in per_block),
        'sum_model_solution_entries': sum(sum((b['model_solution_entries_before_release'] or {}).values())
                                          for b in per_block),
        'sum_zL': sum(b['n_zL'] for b in per_block), 'sum_zU': sum(b['n_zU'] for b in per_block),
        'all_results_succeeded': all(b['solver_result_succeeded'] for b in per_block),
        'all_released': all(b['released'] for b in per_block) if release else None,
        'none_released': (not any(b['released'] for b in per_block)) if not release else None,
        'after_release_all_empty': (all(b['after_release'] == {'model_solutions_n': 0, 'result_solution_n': 0}
                                        for b in per_block) if release else None),
    }
    growth = {k: m_all[k] - m_built[k] for k in ('rss_bytes', 'footprint_bytes') if m_all[k] is not None}
    growth_gc = {k: m_gc[k] - m_built[k] for k in ('rss_bytes', 'footprint_bytes') if m_gc[k] is not None}
    out = {
        'schema': 'p515_s53_w90_b_probe_v1', 'stage': STAGE_TEXT, 'which': which, 'instance': probe_instance,
        'arm': arm, 'release_solution_bookkeeping': release, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'script_sha256': H.sha256_file(os.path.abspath(__file__)), 'harness_sha256': H.sha256_file(H.HARNESS_PATH),
        'case_path': case_rel, 'case_sha256': case_sha,
        'candidate': 'n7_4h_e1 (node 7: 0.25 MVA / 1.0 MWh, 2025) -- the W89 probe candidate',
        'premium_in_force': dict(premium), 'interceptor_calls': interceptor.counts(),
        'method': ('zero-solve reproduction of the bookkeeping objects: NL write (legacy writer call path) -> synthetic '
                   'IPOPT .sol (real layout; zL on finite lower bounds, zU on finite upper bounds with unequal bounds) '
                   '-> Pyomo sol reader with the model IMPORT suffixes -> OptSolver.solve load_solutions=False tail -> '
                   'production model.solutions.load_from -> production replace_warm_start_suffix (zL, zU) -> [ON arm] '
                   'production network._release_solution_bookkeeping'),
        'limitations': [
            'values are synthetic (duals 1e-6, primal 0.5, multipliers +/-1e-6): the STRUCTURE of the objects is '
            'reproduced, not the numbers; object sizes do not depend on the values (one float per entry either way)',
            'the IPOPT process and its workspace, pristine snapshot clones, AA memory, the terminal capture / workbook '
            'and the post-certification step are NOT reproduced: they are (b)-independent, so they do not enter the '
            'saving, but they do enter the runtime peak, which the smoke measures directly',
            'what a release returns to the OS is allocator-dependent; the saving is measured as the difference of the '
            'post-load memory of two fresh processes, each net of its own post-build level, not as an RSS drop',
            'one measurement per arm, no repeats; build-level noise between the two processes is removed by netting',
            'the pickle transient is sampled at 20 ms; W89 found this quantity noisy (+5.27 GiB committed, +6.56 GiB '
            'in an earlier uncommitted run)',
        ],
        'build_wall_s': build_wall, 'loop_wall_s': loop_wall, 'marks': marks,
        'memory_available_at_start': mem_start,
        'models_built': m_built, 'all_loaded': m_all, 'after_gc': m_gc,
        'growth_over_built_bytes': growth, 'growth_over_built_after_gc_bytes': growth_gc,
        'totals': totals, 'per_block': per_block, 'attribution_w32_walk': attribution,
        'pickle_step': pickle_step, 'real_pair_validation': validation,
        'committed_w32_attribution_for_comparison': committed_w32, 'setter_check_release_solution_bookkeeping': setter_check,
        'read_redirected_to': read_dir, 'nl_scratch_dir': nl_dir,
        'solve_claim': 'ZERO SOLVES, guard-verified (every armed permitted=() guard verify(0))',
        'guards': guards_verify(), 'wall_s': time.time() - started,
    }
    os.makedirs(_abs(B_PROBE_DIR_REL), exist_ok=True)
    H._write_once_json(_abs(out_rel), out)
    _log(f"[{tag}] {n_blocks} blocks; sum n {totals['sum_nl_vars']} m {totals['sum_nl_cons']}; growth over built: rss "
         f"{growth.get('rss_bytes', 0) / GIB:.3f} GiB footprint {growth.get('footprint_bytes', 0) / GIB:.3f} GiB "
         f"(after gc {growth_gc.get('rss_bytes', 0) / GIB:.3f} / {growth_gc.get('footprint_bytes', 0) / GIB:.3f})")
    if attribution:
        _log(f"[{tag}] W32 walk: freeable container bytes {attribution['total_freeable_container_bytes'] / GIB:.3f} "
             f"GiB over {attribution['total_entries']} entries")
    if pickle_step:
        _log(f"[{tag}] pickle: {pickle_step.get('status')} "
             f"{ {k: round(pickle_step[k] / GIB, 3) for k in ('transient_rss_sampled_bytes', 'transient_footprint_sampled_bytes', 'transient_ru_maxrss_bytes_W89_definition') if k in pickle_step} } GiB")
    if validation:
        _log(f"[{tag}] real-pair validation: {validation.get('checks')} all={validation.get('all_checks_pass')}")
    if setter_check:
        _log(f"[{tag}] setter check: true took_effect {setter_check['true']['took_effect']}, false took_effect "
             f"{setter_check['false']['took_effect']}")
    _log(f'[{tag}] wrote {out_rel} sha256 {_sha(out_rel)}')
    _finish(0, f'wall={time.time() - started:.1f}s')


# ======================================================================================================================
#  ITEM 1 -- the estimates (from the committed probes; formulas stated operationally and preserved in the spec)
# ======================================================================================================================
ESTIMATE_FORMULAS = {
    'route_A_saving': ('Route A (direct, zero-solve reproduction): saving = (M_off(all loaded) - M_off(built)) - '
                       '(M_on(all loaded) - M_on(built)), M = phys_footprint (primary; it counts compressed pages, the '
                       'W32 reading of "resident") or RSS; the same with M after gc.collect(); each arm net of its own '
                       'post-build level'),
    'route_B_saving': ('Route B (committed evidence scaled by a stated law): saving = E_3x3 x c x r, E_3x3 = the exact '
                       'sum over the 80 blocks of ModelSolution entries (variables incl. fixed + active constraints), '
                       'counted by the OFF probe; c = freeable container bytes per entry from the committed W32 '
                       'attribution (SRP1 and paper blocks: the law "bookkeeping is linear in entries" holds there to '
                       'within c_paper / c_srp1); r = the RSS fraction a release returned on the paper block (W32 '
                       'variants c / e median over the freeable bytes); low = E x min(c) x min(r), high = E x max(c) x 1'),
    'route_W_saving': ('Route W (the committed W32 object walk applied to the 3 x 3 blocks themselves, by import): the '
                       'OFF probe\'s total freeable container bytes (ModelSolution + the SolverResults\' additional '
                       'containers; floats excluded, they are shared with Var values and suffixes)'),
    'saving_estimate': ('central = Route A footprint (after gc); uncertainty = [min, max] over Route A (rss / footprint, '
                        'with / without gc), Route B [low, high] and Route W'),
    'sustained_on': ('SUSTAINED_on = SUSTAINED_off - saving, SUSTAINED_off = the v34 memory model (build_3x3 x k_run; '
                     'range low = build_3x3 x k_run_srp1_low); range = [SUSTAINED_low - saving_max, SUSTAINED_off - '
                     'saving_min]'),
    'smoke_peaks': ('SMOKE_off = the v34 SMOKE (build_3x3 x k_smoke, calibrated on the 2 x 2 cap-2 smoke; cap 3 adds '
                    'one cycle); SMOKE_on = SMOKE_off - saving'),
    'persist_transient': ('T_persist = the MAX of every recorded 3 x 3 pickle transient: the ON probe\'s (sampled RSS, '
                          'sampled footprint, the W89 ru_maxrss definition) and W89\'s (+5.269 GiB committed; +6.562 GiB '
                          'reported in commit 5bc57277\'s message from an earlier uncommitted-revision run) -- the '
                          'conservative choice, it can only err toward NOT persisting'),
    'persist_peak_on': 'PERSIST_on = SUSTAINED_on + T_persist (the transient on top of the peak: an upper bound)',
}


def _probe_rel(which):
    return os.path.join(B_PROBE_DIR_REL, f'b_probe_{which}.json')


def load_probes():
    out, failures = {}, []
    for w in B_PROBES:
        rel = _probe_rel(w)
        if not os.path.isfile(_abs(rel)):
            failures.append(f'b probe {w} missing: {rel}')
            continue
        if not _committed_clean(rel):
            failures.append(f'b probe {w} not committed / clean: {rel}')
        out[w] = {'path': rel, 'sha256': _sha(rel), 'data': _load(rel)}
    return out, failures


def memory_estimates(probes, v34_model):
    off, on, srp1 = (probes[w]['data'] for w in B_PROBES)
    w32 = _w32_committed_srp1_attribution()

    def net(p, key, after):
        return (p['after_gc' if after else 'all_loaded'][key] or 0) - (p['models_built'][key] or 0)

    route_a = {f'{key}{"_after_gc" if g else ""}': net(off, key, g) - net(on, key, g)
               for key in ('footprint_bytes', 'rss_bytes') for g in (False, True)}
    entries = off['totals']['sum_model_solution_entries']
    c = {k: w32[k]['container_bytes_per_entry'] for k in ('srp1', 'paper')}
    paper_freeable_mib = w32['paper']['union_container_bytes'] / MIB
    r = {v: -w32['paper_release_rss_mib'][v]['median'] / paper_freeable_mib for v in ('c', 'e')}
    route_b = {'E_3x3_entries': entries, 'c_bytes_per_entry': c, 'r_rss_fraction_released': r,
               'low_bytes': entries * min(c.values()) * min(r.values()),
               'high_bytes': entries * max(c.values()) * 1.0}
    route_w = (off.get('attribution_w32_walk') or {}).get('total_freeable_container_bytes')
    candidates = list(route_a.values()) + [route_b['low_bytes'], route_b['high_bytes']] + ([route_w] if route_w else [])
    central = route_a['footprint_bytes_after_gc']
    saving = {'central_bytes': central, 'min_bytes': min(candidates), 'max_bytes': max(candidates),
              'central_gib': central / GIB, 'range_gib': [min(candidates) / GIB, max(candidates) / GIB],
              'per_child_prediction_addendum_48_gib': 3.0}
    sus, sus_low = v34_model['sustained_bytes'], v34_model['sustained_range_gib'][0] * GIB
    sustained_on = {'central_bytes': sus - central, 'low_bytes': sus_low - saving['max_bytes'],
                    'high_bytes': sus - saving['min_bytes']}
    smoke_off = v34_model['smoke_bytes']
    smoke_on = {'central_bytes': smoke_off - central, 'low_bytes': smoke_off - saving['max_bytes'],
                'high_bytes': smoke_off - saving['min_bytes']}
    pk = on.get('pickle_step') or {}
    t_on = {k: pk.get(k) for k in ('transient_rss_sampled_bytes', 'transient_footprint_sampled_bytes',
                                   'transient_ru_maxrss_bytes_W89_definition')}
    w89_committed = v34_model['probes']['s53_3x3']['pickle']['transient_over_rss_before_bytes']
    w89_uncommitted = 6.562 * GIB
    t_all = [v for v in t_on.values() if isinstance(v, (int, float))] + [w89_committed, w89_uncommitted]
    t_persist = max(t_all)
    persist_on = {'central_bytes': sustained_on['central_bytes'] + t_persist,
                  'high_bytes': sustained_on['high_bytes'] + t_persist}
    best = v34_model['best_available_observed_bytes']
    gib = lambda d: {k.replace('_bytes', '_gib'): v / GIB for k, v in d.items() if isinstance(v, (int, float))}  # noqa: E731
    return {
        'formulas': ESTIMATE_FORMULAS,
        'route_A_saving_bytes': route_a, 'route_A_saving_gib': gib(route_a),
        'route_B': route_b, 'route_B_gib': [route_b['low_bytes'] / GIB, route_b['high_bytes'] / GIB],
        'route_W_bytes': route_w, 'route_W_gib': (route_w / GIB) if route_w else None,
        'saving_per_child': saving,
        'v34_model_inputs': {k: v34_model[k] for k in ('sustained_bytes', 'sustained_gib', 'sustained_range_gib',
                                                       'smoke_bytes', 'smoke_gib', 'persist_peak_gib', 'k_run',
                                                       'k_run_srp1_low', 'k_smoke', 'f_term', 'k_tr',
                                                       'best_available_observed_gib', 'hw_memsize_bytes')},
        'sustained_off_gib': sus / GIB, 'sustained_on': sustained_on, 'sustained_on_gib': gib(sustained_on),
        'smoke_off_gib': smoke_off / GIB, 'smoke_on': smoke_on, 'smoke_on_gib': gib(smoke_on),
        'persist_transient': {'on_probe_bytes': t_on, 'w89_committed_bytes': w89_committed,
                              'w89_uncommitted_reported_bytes': w89_uncommitted,
                              'w89_uncommitted_source': 'git commit 5bc57277 message (earlier uncommitted-revision run)',
                              'T_persist_bytes': t_persist, 'T_persist_gib': t_persist / GIB},
        'persist_peak_on': persist_on, 'persist_peak_on_gib': gib(persist_on),
        'persist_peak_off_v34_gib': v34_model['persist_peak_gib'],
        'margin_rule_preview_against_best_recorded_availability': {
            'best_recorded_available_gib': best / GIB, 'threshold_gib': MARGIN * best / GIB,
            'persist_on_central_gib': persist_on['central_bytes'] / GIB,
            'would_persist': persist_on['central_bytes'] <= MARGIN * best,
            'note': 'a PREVIEW only; the rule uses the smoke\'s measured peak and the post-reboot availability'},
        'g_growth_cap_to_full_run': v34_model['k_run'] / v34_model['k_smoke'],
        'srp1_validation': {'real_pair': srp1.get('real_pair_validation'),
                            'setter_check_took_effect': {k: (srp1.get('setter_check_release_solution_bookkeeping') or {})
                                                         .get(k, {}).get('took_effect') for k in ('true', 'false')},
                            'committed_w32': w32},
    }


# ======================================================================================================================
#  G6 v35 -- the final scope, the tier-2 non-vacuity ruling, self-tests
# ======================================================================================================================
G6_V35 = {
    'name': 'G6_floor_records_v35',
    'replaces': 'v32 G6_floor_records_v32 (v34 applied it at B = 80)',
    'FINAL_SCOPE_for_every_future_cell': (
        'population P = every persisted network_ipopt_solve_records.jsonl record of the tail window W (the cycles the '
        'convergence-depth tail was ACTIVE, convergence_depth_tail_state.json per_cycle active) and of the terminal '
        'round T (= cycles_run); every record of P is judged on the per-record predicate; the terminal round is judged '
        'on the FINAL ACCEPTED ATTEMPT PER BLOCK of T (superseded attempts counted and reported, never judged on floor '
        'status); records outside P are reported, never judged'),
    'B_blocks_per_round': X.G6_V32['B_blocks_per_round'],
    'block': X.G6_V32['block'], 'ladder': X.G6_V32['ladder'], 'final_attempt': X.G6_V32['final_attempt'],
    'superseded_attempts': X.G6_V32['superseded_attempts'], 'accepted': X.G6_V32['accepted'],
    'final_accepted_attempt': X.G6_V32['final_accepted_attempt'],
    'predicate_every_record_of_P': X.G6_V32['predicate_every_record_of_P'],
    'predicate_terminal_block': X.G6_V32['predicate_terminal_block'],
    'tier2_terminal_blocks_RULING': (
        'PLANNER RULING (W89 question 4), fixed BEFORE any 3 x 3 run: a block whose final accepted attempt in T is a '
        'tier-2 adaptive-mu retry (floor formula not applicable) does NOT fail G6 through non-vacuity. It is judged on '
        'the conditions that apply -- the per-record predicate: compl_inf_tol_in_force == 1e-6 in W u {T}, '
        'options_list_agrees, no residual parse reason beyond the declared ruling-2 class -- and COUNTED '
        '(terminal_round.n_judged_not_applicable, judged_not_applicable listed)'),
    'non_vacuity': ('v30 part verbatim (P non-empty; every round of W u {T} holds exactly B primary-attempt records and '
                    '>= B records) AND T holds exactly B blocks AND every block of T has exactly one judged attempt AND '
                    'exactly B blocks of T have an ACCEPTED judged (final) attempt [v35: "exactly one judged accepted '
                    'attempt per block"; replaces v32\'s ">= B judged APPLICABLE attempts in T", which failed a tier-2 '
                    'terminal block by construction; the applicable count is REPORTED, not gated]'),
    'passes_iff': 'non_vacuity holds AND every record of P satisfies its predicate AND every block of T satisfies its',
    'floor_status_definition': X.G6_V32['floor_status_definition'],
    'not_judged': X.G6_V32['not_judged'],
    'design_consequence_recorded': (
        'with no lower bound on judged APPLICABLE attempts, a terminal round in which EVERY block\'s final accepted '
        'attempt were a tier-2 retry would PASS with zero floor tests (self-test V5 declares exactly this); the '
        'applicable / not-applicable counts are therefore reported next to the verdict on every cell, and a terminal '
        'round with any tier-2 final attempt is flagged in the pair results for the Planner'),
    'computation': ('p515_s53_w90_3x3_campaign.g6_v35_evaluate_records: the committed v32 evaluator '
                    '(p515_s53_w89_g6_final_attempt_reeval.g6_v32_evaluate_records, by import) for every per-record and '
                    'per-block judgement, with the non-vacuity recomputed as above; the v32 non-vacuity is kept in the '
                    'output, reported, not gated'),
    'timing': 'fixed in v35 BEFORE any 3 x 3 evaluation exists: NOT post-hoc',
}
G6_POST_HOC_RECORD = {
    'standing_rule': ('a gate\'s scope is part of the frozen spec, fixed before the run (Addendum 48 restates it); the '
                      'v35 scope above is written for every FUTURE cell'),
    'v30_POST_HOC': {'spec': 'frozen_s53_spec_v30_bb6703da.json', 'frozen_utc': '2026-09-25T17:46:23Z',
                     'what': ('G6 re-scoped from EVERY record of the run (v29) to the tail window W and terminal round '
                              'T, AFTER the W86 tight-tail re-certification had run and failed v29\'s G6 on 3 pre-tail '
                              'tier-2 records of C* (rounds 11 / 13 / 23)'),
                     'status': 'POST-HOC: the scope was changed after the run it judged'},
    'v31_post_run_strengthening': {'spec': 'frozen_s53_spec_v31_cded3496.json', 'frozen_utc': '2026-09-25T18:02:29Z',
                                   'what': ('added the terminal-round floor test and ruling 2 (tier-2 records judged on '
                                            'the conditions that apply), after the same run'),
                                   'status': 'also frozen after the run; a strengthening, not a narrowing'},
    'v32_POST_HOC': {'spec': 'frozen_s53_spec_v32_69449731.json', 'frozen_utc': '2026-09-25T18:15:46Z',
                     'what': ('judged the FINAL ACCEPTED attempt per block of T instead of every applicable T record, '
                              'AFTER the W86 run (the W88 question 1: a retried primary above the floor would have '
                              'failed v31)'),
                     'status': 'POST-HOC: the scope was changed after the run it judged'},
    'acceptance_of_the_tail_rests_on': ('the three measured facts of Addendum 48 (re-certification at identical cycles; '
                                        'per-cycle gross bitwise before the first tail cycle; 48 / 48 terminal solves '
                                        'at the mu floor with 1e-6 in force), not on G6'),
    'v35_tier2_ruling': 'fixed BEFORE any 3 x 3 run -- NOT post-hoc',
}


def g6_v35_evaluate_records(records, window, terminal, prod, b):
    g = X.g6_v32_evaluate_records(records, window, terminal, prod, b)
    nv32 = g['non_vacuity']
    t = g['terminal_round']
    n_accepted_final = sum(v for k, v in t['judged_exit'].items() if k in X.ACCEPTED_EXITS)
    nv = {'v30_part': nv32['v30_part'], 'terminal_blocks_eq_B': nv32['terminal_blocks_eq_B'],
          'one_judged_attempt_per_terminal_block': nv32['one_judged_attempt_per_terminal_block'],
          'terminal_judged_accepted_eq_B': n_accepted_final == b}
    nv['holds'] = all(nv.values())
    out = dict(g)
    out['gate_version'] = 'v35'
    out['non_vacuity'] = nv
    out['non_vacuity_v32_reported_not_gated'] = nv32
    out['n_judged_accepted_final'] = n_accepted_final
    out['tier2_terminal_blocks_counted'] = {'n_judged_not_applicable': t['n_judged_not_applicable'],
                                            'n_judged_applicable': t['n_judged_applicable'],
                                            'judged_not_applicable': t['judged_not_applicable']}
    out['gate_pass'] = nv['holds'] and not g['bad_records'] and not g['bad_blocks']
    out['gate_pass_v32_reported'] = g['gate_pass']
    return out


def g6_v35_evaluate(eval_dir, rec, b):
    records, window, terminal, prod = X.cell_inputs(eval_dir, rec)
    return X._ladders_json(g6_v35_evaluate_records(records, window, terminal, prod, b))


G6_V35_SELF_TESTS = [
    {'id': 'V1', 'transform': ('RULING: the real round-11 case33_3/2035/Winter ladder of C* (primary max_iter above, '
                               'recovery max_iter above, tier-2 Optimal, floor not applicable) transplanted into T in '
                               'place of that block\'s T primary (round -> T, compl_inf_tol_in_force -> 1e-6) [= v32 T6]'),
     'expect_pass': True, 'expect_bad_records': 0, 'expect_block_failures': [], 'expect_nv_false': [],
     'expect_n_not_applicable': 1, 'expect_v32_pass': False,
     'why': 'the Planner ruling: a tier-2 final accepted attempt passes, counted; v32 FAILED it through non-vacuity'},
    {'id': 'V2', 'transform': 'as V1, the tier-2 record\'s compl_inf_tol_in_force -> 1e-4',
     'expect_pass': False, 'expect_bad_records': 1, 'expect_bad_record_failures': ['compl_inf_tol_in_force'],
     'expect_block_failures': [], 'expect_nv_false': [], 'expect_n_not_applicable': 1,
     'why': 'a tier-2 terminal block is still judged on the tolerance condition'},
    {'id': 'V3', 'transform': 'as V1, the tier-2 record\'s options_list_agrees -> False',
     'expect_pass': False, 'expect_bad_records': 1, 'expect_bad_record_failures': ['options_list_agrees'],
     'expect_block_failures': [], 'expect_nv_false': [], 'expect_n_not_applicable': 1,
     'why': 'a tier-2 terminal block is still judged on the options-list condition'},
    {'id': 'V4', 'transform': 'NEGATIVE CONTROL: every record of the terminal round T deleted (a vacuous terminal round)',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': [],
     'expect_nv_false': ['v30_part', 'terminal_blocks_eq_B', 'one_judged_attempt_per_terminal_block',
                         'terminal_judged_accepted_eq_B'],
     'why': 'a genuinely vacuous terminal round still FAILS non-vacuity'},
    {'id': 'V5', 'transform': ('RECORDED CONSEQUENCE: EVERY T primary replaced by a transplanted tier-2 ladder (the '
                               'round-11 ladder\'s three records re-labelled to each block, round -> T, tolerance 1e-6)'),
     'expect_pass': True, 'expect_bad_records': 0, 'expect_block_failures': [], 'expect_nv_false': [],
     'expect_n_not_applicable': 'B', 'expect_n_applicable': 0,
     'why': ('under the ruling a terminal round with zero floor tests passes; the counts are reported on every cell -- '
             'recorded so the consequence is not discovered later')},
    {'id': 'V6', 'transform': 'NEGATIVE CONTROL: one T primary deleted (T holds B - 1 blocks) [= v32 T17]',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': [],
     'expect_nv_false': ['v30_part', 'terminal_blocks_eq_B', 'terminal_judged_accepted_eq_B'],
     'why': 'a missing block fails non-vacuity'},
    {'id': 'V7', 'transform': 'none (positive control) [= v32 T7]', 'expect_pass': True, 'expect_bad_records': 0,
     'expect_block_failures': [], 'expect_nv_false': [], 'expect_n_not_applicable': 0,
     'why': 'the real C* cell passes'},
    {'id': 'V8', 'transform': ('the T primary -> max_iter above; recovery max_iter above; the tier-2 retry (copy of the '
                               'real round-11 tier-2, round T, 1e-6) -> exit max_iter [= v32 T13]'),
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['no_accepted_attempt'],
     'expect_nv_false': ['terminal_judged_accepted_eq_B'],
     'why': 'a block with NO accepted attempt fails, and it is not an accepted judged attempt for non-vacuity'},
    {'id': 'V9', 'transform': 'one T primary (no retry) floor_status -> "above", mu_over_floor 7.0 [= v32 T8]',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['final_accepted_attempt_floor_not_at'],
     'expect_nv_false': [], 'why': 'an applicable final accepted attempt above the floor fails'},
    {'id': 'V10', 'transform': ('the T primary -> max_iter above; a recovery appended (Optimal, floor at) [= v32 T12]'),
     'expect_pass': True, 'expect_bad_records': 0, 'expect_block_failures': [], 'expect_nv_false': [],
     'why': 'only a superseded attempt above the floor: passes'},
    {'id': 'V11', 'transform': 'the T primary -> exit "Converged to a point of local infeasibility..." (no retry) [= T14]',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['no_accepted_attempt'],
     'expect_nv_false': ['terminal_judged_accepted_eq_B'],
     'why': 'a single-attempt block whose only attempt was not accepted fails'},
]


def g6_v35_self_tests():
    """G6_V35_SELF_TESTS on the REAL C* records of the W86 tight-tail re-certification (deep copies), the v32 self-test
    constructions (p515_s53_w89_g6_final_attempt_reeval.self_tests) re-stated here for the v35 non-vacuity."""
    entries = X.W._cells_from_recert_spec()
    d = os.path.join(X.W.RECERT_ROOT, 'evals', entries['c_star']['eval_dir'])
    rec = X._load(os.path.join(d, 'evaluation_record.json'))
    records, window, terminal, prod = X.cell_inputs(d, rec)
    b = X.SRP1_BLOCKS_PER_ROUND
    t2 = [r for r in records if r.get('round') == 11 and r.get('attempt') == 'recovery_tier2']
    lad11 = sorted([r for r in records if t2 and X.block_of(r) == X.block_of(t2[0])],
                   key=lambda a: X.LADDER.index(a['attempt']))
    tp_all = [r for r in records if r.get('round') == terminal and r.get('attempt') == 'primary'
              and r.get('floor_status') == 'at' and X.accepted(r)]
    found = {'tier2_round11': len(t2), 'round11_ladder_len': len(lad11), 'terminal_primary_at_accepted': len(tp_all)}
    if len(t2) != 1 or len(lad11) != 3 or len(tp_all) != b:
        return {'base_records_found': found}, False
    tgt = next(r for r in tp_all if X.block_of(r) == (terminal,) + X.block_of(t2[0])[1:])

    def replace(pop, mapping):
        out = []
        for r in pop:
            hit = next((new for old, new in mapping if r is old), None)
            out.extend(hit if hit is not None else [r])
        return out

    def ladder_into(block_rec, tol2=X.TAIL_TOL, opts2=True):
        moved = []
        for r in lad11:
            m = copy.deepcopy(r)
            m['round'] = terminal
            for k in ('network', 'year', 'day'):
                m[k] = block_rec.get(k)
            m['compl_inf_tol_in_force'] = X.TAIL_TOL
            if m['attempt'] == 'recovery_tier2':
                m['compl_inf_tol_in_force'] = tol2
                if not opts2:
                    m['options_list_agrees'] = False
            moved.append(m)
        return moved

    def failed_primary(mu=7.0):
        p = copy.deepcopy(tgt)
        p['exit'] = 'Maximum Number of Iterations Exceeded.'
        p['floor_status'] = 'above'
        p['mu_over_floor'] = mu
        return p

    def retry(base, attempt, exit_, floor, mu):
        r = copy.deepcopy(base)
        r.update({'attempt': attempt, 'warm_start': False, 'exit': exit_, 'floor_status': floor, 'mu_over_floor': mu})
        return r

    def population(tid):
        pop = list(records)
        if tid == 'V1':
            return replace(pop, [(tgt, ladder_into(tgt))])
        if tid == 'V2':
            return replace(pop, [(tgt, ladder_into(tgt, tol2=1e-4))])
        if tid == 'V3':
            return replace(pop, [(tgt, ladder_into(tgt, opts2=False))])
        if tid == 'V4':
            return [r for r in pop if r.get('round') != terminal]
        if tid == 'V5':
            return replace(pop, [(p, ladder_into(p)) for p in tp_all])
        if tid == 'V6':
            return replace(pop, [(tgt, [])])
        if tid == 'V7':
            return pop
        if tid == 'V8':
            p = failed_primary()
            r1 = retry(p, 'recovery', 'Maximum Number of Iterations Exceeded.', 'above', 5.0)
            r2 = copy.deepcopy(t2[0])
            r2.update({'round': terminal, 'compl_inf_tol_in_force': X.TAIL_TOL,
                       'exit': 'Maximum Number of Iterations Exceeded.'})
            return replace(pop, [(tgt, [p, r1, r2])])
        if tid == 'V9':
            p = copy.deepcopy(tgt)
            p['floor_status'] = 'above'
            p['mu_over_floor'] = 7.0
            return replace(pop, [(tgt, [p])])
        if tid == 'V10':
            p = failed_primary()
            return replace(pop, [(tgt, [p, retry(p, 'recovery', 'Optimal Solution Found.', 'at', 1.0)])])
        if tid == 'V11':
            p = copy.deepcopy(tgt)
            p['exit'] = 'Converged to a point of local infeasibility. Problem may be infeasible.'
            return replace(pop, [(tgt, [p])])
        raise KeyError(tid)

    out, ok = [], True
    for t in G6_V35_SELF_TESTS:
        pop = population(t['id'])
        g = g6_v35_evaluate_records(pop, window, terminal, prod, b)
        nv_false = sorted(k for k, v in g['non_vacuity'].items() if k != 'holds' and not v)
        block_failures = sorted({f for bb in g['bad_blocks'] for f in bb['failures']})
        bad_rec_failures = sorted({f for x in g['bad_records'] for f in x['failures']})
        tr = g['terminal_round']
        checks = {'pass': g['gate_pass'] == t['expect_pass'],
                  'bad_records': g['n_bad_records'] == t['expect_bad_records'],
                  'block_failures': block_failures == sorted(t['expect_block_failures']),
                  'non_vacuity_components_false': nv_false == sorted(t['expect_nv_false'])}
        if 'expect_bad_record_failures' in t:
            checks['bad_record_failures'] = all(f in bad_rec_failures for f in t['expect_bad_record_failures'])
        if 'expect_n_not_applicable' in t:
            want = b if t['expect_n_not_applicable'] == 'B' else t['expect_n_not_applicable']
            checks['n_not_applicable'] = tr['n_judged_not_applicable'] == want
        if 'expect_n_applicable' in t:
            checks['n_applicable'] = tr['n_judged_applicable'] == t['expect_n_applicable']
        if 'expect_v32_pass' in t:
            checks['v32_verdict'] = g['gate_pass_v32_reported'] == t['expect_v32_pass']
        res = {'id': t['id'], 'observed_pass': g['gate_pass'], 'expect_pass': t['expect_pass'],
               'observed_v32_pass': g['gate_pass_v32_reported'], 'observed_n_bad_records': g['n_bad_records'],
               'observed_bad_record_failures': bad_rec_failures, 'observed_block_failures': block_failures,
               'observed_non_vacuity_false': nv_false,
               'observed_terminal': {k: tr[k] for k in ('n_blocks', 'n_judged', 'n_judged_applicable',
                                                        'n_judged_not_applicable', 'judged_attempt_labels')},
               'observed_n_judged_accepted_final': g['n_judged_accepted_final'], 'checks': checks,
               'ok': all(checks.values())}
        ok = ok and res['ok']
        out.append(res)
    return {'base': {'cell': 'c_star (W86 tight-tail re-certification)', 'eval_dir': d, 'B': b, 'terminal': terminal,
                     'window': sorted(window), 'found': found}, 'results': out}, ok


# ======================================================================================================================
#  identity_holds -- recomputed under the current formula (Addendum 48: no False flag travels without its explanation)
# ======================================================================================================================
S47_IDENTITY_LOOK = os.path.join(_P53, 's47_identity_look_w88', 's47_identity_look.json')


def solve_identity_recomputed():
    """For every evaluation whose value enters v35's reference chain: the recorded `solve_profile.identity_holds` next
    to the flag RECOMPUTED under the current W35 per-event formula (G.run_admm_arm: observed == (1 + n_dso) x n_years x
    n_days + n_esso per round x (cycles_run + 1) + every retry attempted; unsupported if an ESSO recovery event or an
    'indeterminate' network event exists), from the committed records and failure-event files. The s47 re-certification
    (R_ref pre-tail, superseded) recorded False under the pre-W35 formula `51 * len(rows) + 51` (no retry term)."""
    look = _load(S47_IDENTITY_LOOK)
    out = {'decision': 'RECOMPUTE (not drop)',
           'why': ('dropping the flag would remove a recorded value; recomputing it under the formula every later '
                   'evaluation -- and G5 of this spec -- uses shows it True, with the stale value and its formula kept '
                   'beside it, so the False flag never travels without its explanation'),
           'current_formula': ('observed permitted_solve == base + sum over network-failure events of '
                               '[recovery_attempted] + [tier2_attempted], base = solves_per_cycle x (cycles_run + 1), '
                               'solves_per_cycle = (1 + n_dso) x n_years x n_days + n_esso_nodes; unsupported (-> False) '
                               'when an ESSO recovery event or an indeterminate network event exists'),
           's47_identity_look': {'path': S47_IDENTITY_LOOK, 'sha256': _sha(S47_IDENTITY_LOOK),
                                 **_git_state(S47_IDENTITY_LOOK)},
           'cells': {}}
    for label, c in look['cells'].items():
        d = c['eval_dir']
        rec = _load(os.path.join(d, 'evaluation_record.json'))
        events = _read_jsonl(_abs(os.path.join(d, 'network_failures_s39_D.jsonl')))
        esso_path = _abs(os.path.join(d, 'esso_recovery_events_s39_D.jsonl'))
        esso = _read_jsonl(esso_path) if os.path.isfile(esso_path) else []
        spc = c['per_event_rule']['solves_per_cycle_from_case_file']
        base = spc * ((rec.get('cycles_run') or 0) + 1)
        retries = sum(int(bool(e.get('recovery_attempted'))) + int(bool(e.get('tier2_attempted'))) for e in events)
        supported = not esso and not any(e.get('class') == 'indeterminate' for e in events)
        observed = (rec.get('solve_profile') or {}).get('observed', {}).get('permitted_solve')
        out['cells'][f's47_recert:{label}'] = {
            'eval_dir': d, 'recorded_identity_holds': (rec.get('solve_profile') or {}).get('identity_holds'),
            'recorded_formula': c['pre_W35_formula'], 'recorded_formula_reproduces_flag':
                c['pre_W35_formula_reproduces_recorded_flag'],
            'observed': observed, 'base': base, 'retries_attempted': retries, 'supported': supported,
            'recomputed_identity_holds': bool(supported and observed == base + retries),
            'agrees_with_w88_look': c['per_event_rule']['holds'] == bool(supported and observed == base + retries)}
    w86 = _load(os.path.join(_P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'campaign_results.json'))
    for label, c in w86['per_cell'].items():
        sp = _load(os.path.join(c['eval_dir'], 'evaluation_record.json')).get('solve_profile') or {}
        out['cells'][f'w86_tail_recert:{label}'] = {'eval_dir': c['eval_dir'],
                                                    'recorded_identity_holds': sp.get('identity_holds'),
                                                    'recorded_under_current_formula': True,
                                                    'expected': sp.get('expected_solves'),
                                                    'observed': (sp.get('observed') or {}).get('permitted_solve')}
    out['all_recomputed_true'] = all(v.get('recomputed_identity_holds', v.get('recorded_identity_holds')) is True
                                     for v in out['cells'].values())
    return out


# ======================================================================================================================
#  the harness plumbing for (b): zero-solve checks
# ======================================================================================================================
def harness_b_plumbing_checks(derived):
    import network as NET
    child_src = inspect.getsource(H._child_real)
    hook_src = inspect.getsource(H._config_hook_factory)
    key_src = inspect.getsource(H.evaluation_key)
    freeze_src = inspect.getsource(H.freeze_campaign_spec)
    run_src = inspect.getsource(NET._run_smopf)
    src = {
        'option_key_declared': 'release_solution_bookkeeping' in H.EVALUATION_OPTION_KEYS,
        'child_reads_the_entry_option': "validate_release_solution_bookkeeping(entry.get('release_solution_bookkeeping'))"
                                        in child_src,
        'child_passes_it_to_the_config_hook': 'release_solution_bookkeeping=release_bk' in child_src,
        'child_installs_the_call_counter': '_ReleaseBookkeepingCallCounter(holder) if release_bk is not None' in child_src,
        'hook_applies_it_with_the_srp1_gate_setter': 'S44.set_release_solution_bookkeeping(planning, release_bk)'
                                                     in hook_src,
        'evaluation_key_never_reads_it': 'release_solution_bookkeeping' not in key_src,
        'freeze_does_not_pass_it_to_the_key': ('release_bk = validate_release_solution_bookkeeping' in freeze_src
                                               and 'release_bk' not in freeze_src.split('ekey = evaluation_key(')[1]
                                               .split(')')[0]),
        'production_run_smopf_calls_release_under_the_switch': (
            "getattr(params.solver_params, 'release_solution_bookkeeping', False)" in run_src
            and '_release_solution_bookkeeping(model, result)' in run_src),
    }
    # K1: three scratch campaign freezes of x0 (option absent / True / False): identical eval keys == v34's key
    keys, entries = {}, {}
    for tag, opt in (('absent', None), ('true', True), ('false', False)):
        root = tempfile.mkdtemp(prefix=f'w90_k1_{tag}_')
        options = {'investment_year': W9.YEAR, 'interface_deviation_premium': dict(W9.PREMIUM)}
        if opt is not None:
            options['release_solution_bookkeeping'] = opt
        try:
            _p, _sha256, spec = H.freeze_campaign_spec(
                os.path.join(root, 'c'), f'w90_k1_{tag}', [('x0', W9._nodes('x0'), options)],
                configuration={'name': 'W90 K1 scratch', 'arm_label': W9.ARM_LABEL, 'overrides': {},
                               'case_file_anderson_acceleration': dict(W9.CASE_FILE_AA),
                               'ess_ageing_baseline': copy.deepcopy(W9.ESS_AGEING_BASELINE),
                               'ess_ageing_baseline_label': W9.LABEL, 'derived_instance': derived,
                               'convergence_depth_tail': dict(W9.TAIL)},
                cap=3, concurrency=1, authority=['W90 K1 scratch check'], required_consecutive_cycles=10)
            e = spec['candidates'][0]
            keys[tag] = e['eval_key']
            entries[tag] = {'has_option_field': 'release_solution_bookkeeping' in e,
                            'value': e.get('release_solution_bookkeeping'), 'entry_keys': sorted(e)}
        finally:
            shutil.rmtree(root, ignore_errors=True)
    refused = False
    try:
        H.validate_release_solution_bookkeeping(1)
    except ValueError:
        refused = True
    # K4: the counter is pass-through, counts, and restores the production function
    import pyomo.environ as pe
    holder = {}
    original = NET._release_solution_bookkeeping

    class _Res:
        class _Sol:
            def __init__(self):
                self.cleared = 0

            def clear(self):
                self.cleared += 1
        solution = _Sol()

    m = pe.ConcreteModel()
    with H._ReleaseBookkeepingCallCounter(holder):
        wrapped_in_place = NET._release_solution_bookkeeping is not original
        NET._release_solution_bookkeeping(m, _Res)
    counter = {'wrapped_in_place': wrapped_in_place, 'count': holder.get('release_solution_bookkeeping_calls'),
               'restored': NET._release_solution_bookkeeping is original, 'result_solution_cleared_once':
                   _Res.solution.cleared == 1}
    v34_x0 = _load(SPEC_V34['path'])['cells']['x0']['eval_key']
    k1 = {'keys': keys, 'entries': entries, 'all_three_keys_equal': len(set(keys.values())) == 1,
          'equal_to_v34_x0_key': set(keys.values()) == {v34_x0}, 'v34_x0_key': v34_x0,
          'absent_entry_has_no_field': not entries['absent']['has_option_field'],
          'true_false_recorded': (entries['true']['value'] is True and entries['false']['value'] is False),
          'non_bool_refused': refused}
    checks = {**{f'source:{k}': v for k, v in src.items()},
              **{f'K1:{k}': v for k, v in k1.items() if isinstance(v, bool)},
              **{f'K4:{k}': v for k, v in counter.items() if isinstance(v, bool)},
              'K4:count_is_1': counter['count'] == 1}
    return {'source': src, 'K1_scratch_freezes': k1, 'K4_counter': counter, 'checks': checks,
            'all_hold': all(checks.values()),
            'enters_evaluation_key': False if (src['evaluation_key_never_reads_it'] and k1['all_three_keys_equal'])
            else 'CHECK FAILED'}


# ======================================================================================================================
#  keys, pre-launch assertion, rule eleven
# ======================================================================================================================
def pre_launch_assertion(derived, specs=()):
    """Every pair cell's eval key, recomputed now: (a) EQUALS the v34 frozen key (carried over -- option (b) is not
    keyed); (b) differs from the pre-tail 3 x 3 key and the declared-OFF key equals it (W86 K-check form); (c) differs
    from the 2 x 2 alpha-row key of the same candidate; (d) is absent from every committed campaign spec outside the
    w89_3x3 and w90_3x3 roots (the superseded v33 / v34 freezes and the v35 arms share the keys by design); and every
    given frozen campaign spec's entries carry exactly the recomputed keys."""
    own = (W9.ROOT_REL, ROOT_REL)
    committed = L.committed_eval_keys(exclude_roots=own)
    v34 = _load(SPEC_V34['path'])
    alpha_pair1 = _load(W9.ALPHA_ROW_PAIR1['path'])
    per = {}
    for label in STAGES['pair']['labels']:
        now = W9._eval_key(label, derived)
        pre_tail = W9._eval_key(label, derived, tail=None)
        declared_off = W9._eval_key(label, derived, tail={'enabled': False, 'compl_inf_tol': 1e-6})
        a_label = {'x0': 'x0_a0p50', 'n7_4h_e1': 'n7_4h_e1_a0p50'}[label]
        a_dir = (alpha_pair1['points'].get(a_label) or {}).get('eval_dir') or ''
        a_key = _load(os.path.join(a_dir, 'evaluation_record.json')).get('eval_key') if a_dir else None
        in_specs = []
        for s in specs:
            for e in s.get('candidates') or []:
                if e['label'] == label:
                    in_specs.append(e.get('eval_key') == now)
        per[label] = {
            'eval_key': now, 'v34_frozen_eval_key': v34['cells'][label]['eval_key'],
            'equals_v34_frozen_key': now == v34['cells'][label]['eval_key'],
            'pre_tail_3x3_key': pre_tail, 'differs_from_pre_tail_3x3_key': now != pre_tail,
            'declared_off_key_equals_pre_tail_key': declared_off == pre_tail,
            'alpha_row_2x2_key_same_candidate': a_key,
            'differs_from_2x2_alpha_row_key': a_key is not None and now != a_key,
            'absent_from_every_committed_spec_outside_w89_w90': now not in committed,
            'committed_specs_holding_it_outside_w89_w90': committed.get(now, []),
            'frozen_entries_equal_recomputed': all(in_specs), 'n_frozen_entries_checked': len(in_specs)}
    holds = all(v['equals_v34_frozen_key'] and v['differs_from_pre_tail_3x3_key']
                and v['declared_off_key_equals_pre_tail_key'] and v['differs_from_2x2_alpha_row_key']
                and v['absent_from_every_committed_spec_outside_w89_w90'] and v['frozen_entries_equal_recomputed']
                for v in per.values())
    return {'per_cell': per, 'holds': holds, 'n_committed_keys_scanned': len(committed), 'excluded_roots': list(own)}


def launcher_checklist():
    """Rule eleven: every quantity v35 requires has a capture path, asserted before any run."""
    base = W9.launcher_checklist()
    child_src = inspect.getsource(H._child_real)
    checks = {
        'b_option_reaches_the_child_and_is_recorded': (
            "'release_solution_bookkeeping_applied_in_child': holder.get('release_solution_bookkeeping_applied')"
            in child_src and "'release_solution_bookkeeping_calls': holder.get('release_solution_bookkeeping_calls')"
            in child_src),
        'child_peak_rss_recorded': "'child_python_process_ru_maxrss': self_ru.ru_maxrss" in child_src,
        'per_cycle_rss_captured': all(f in H.PER_CYCLE_RESPONSE_FIELDS for f in ('rss_bytes', 'ru_maxrss_bytes')),
        'g6_v35_uses_the_v32_block_rules': 'X.g6_v32_evaluate_records(' in inspect.getsource(g6_v35_evaluate_records),
        'bitwise_compare_defined': 'NON_DETERMINISTIC_CYCLE_FIELDS' in inspect.getsource(bitwise_compare),
        'margin_rule_defined': 'MARGIN *' in inspect.getsource(margin_rule),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W90 launcher): capture paths missing: {missing}')
    return {'w90_launcher': checks, 'w89_launcher': base}


# ======================================================================================================================
#  the smoke gate: per-arm checks, bitwise comparison, margin rule
# ======================================================================================================================
def successful_network_solves(records):
    """Blocks (over every round) whose FINAL attempt is accepted -- what `_run_smopf` loads, hence the number of
    `_release_solution_bookkeeping` calls with (b) on (production's load_from blind spot noted in v32's cross-check)."""
    by = {}
    for r in records:
        by.setdefault(X.block_of(r), []).append(r)
    n = 0
    for attempts in by.values():
        final = sorted(attempts, key=lambda a: X.LADDER.index(a['attempt']) if a.get('attempt') in X.LADDER else 99)[-1]
        n += int(X.accepted(final))
    return n, len(by)


def b_plumbing_check(rec, records, expected):
    applied = rec.get('release_solution_bookkeeping_applied_in_child') or {}
    n_ok, n_blocks = successful_network_solves(records)
    calls = rec.get('release_solution_bookkeeping_calls')
    parts = {'declared_in_record': rec.get('release_solution_bookkeeping') is expected,
             'took_effect': applied.get('took_effect') is True and applied.get('requested') is expected,
             'read_back_all': bool(applied.get('read_back')) and all(v is expected for v in
                                                                     applied['read_back'].values()),
             'calls_as_expected': calls == (n_ok if expected else 0)}
    return all(parts.values()), {'parts': parts, 'calls': calls, 'successful_network_solves': n_ok,
                                 'n_blocks_over_all_rounds': n_blocks, 'applied': applied}


def smoke_arm_checks(arm, entry, eval_dir, ss):
    """S1-S13 and S15 on one arm (files only); S14, S16, S17 are cross-arm / parent."""
    cap = STAGES[arm]['cap']
    rounds = cap + 1
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    checks, detail = {}, {'exit_code': exit_code}
    checks['S1_child_exit0_record_uncertified_cap_cycles'] = (
        exit_code == 0 and bool(rec) and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json'))
        and rec.get('status') == 'not_certified' and rec.get('cycles_run') == cap)
    if not rec:
        return checks, detail, rec
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'], expect_certified=False)
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d}
    checks['S2_append_byte_identical'] = c.get('append_reconciles_byte_identical', False)
    checks['S3_checklist_line1_before_any_solve'] = c.get('checklist_line1_before_any_solve', False)
    checks['S4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    ok5, detail['S5'] = W9._solve_profile_check(rec, len(records))
    checks['S5_solve_profile_base_plus_retries'] = ok5 and (rec.get('solve_profile') or {}).get(
        'base_solves') == W9.SOLVES_PER_ROUND * rounds
    events = _read_jsonl(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE))
    ts = json.load(open(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE)))
    applies = [e for e in events if e.get('event') == 'apply']
    nexts = [e for e in events if e.get('event') == 'next_state']
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    stdout_text = open(os.path.join(eval_dir, 'child_stdout.log')).read()
    per_cycle = ts.get('per_cycle') or []
    detail['S6'] = {'apply_cycles': [e.get('cycle') for e in applies],
                    'apply_active': [e['record'].get('active') for e in applies],
                    'next_state_values': [e.get('value') for e in nexts],
                    'row_cycle_convergence': [r.get('cycle_convergence') for r in rows],
                    'tail_on_lines_in_child_stdout': stdout_text.count('Convergence-depth tail ON')}
    checks['S6_tail_inactive_at_cap'] = (
        len(per_cycle) == cap and not any(p.get('active') or p.get('acted') for p in per_cycle)
        and (ts.get('restore_at_exit') or {}).get('acted') is False
        and detail['S6']['apply_cycles'] == list(range(1, cap + 1)) + [None]
        and not any(detail['S6']['apply_active']) and detail['S6']['next_state_values'] == [False] * cap
        and not any(detail['S6']['row_cycle_convergence']) and detail['S6']['tail_on_lines_in_child_stdout'] == 0)
    prod = L._production_compl_inf_tol(ts.get('baseline'))
    bad = [{**X._short(r), 'failures': X.per_record_failures(r, set(), prod)} for r in records
           if X.per_record_failures(r, set(), prod)]
    detail['S7'] = {'n_records': len(records), 'n_bad': len(bad), 'bad_first': bad[:10], 'production': prod,
                    'records_per_round': dict(sorted(Counter(r.get('round') for r in records).items())),
                    'floor_status_tally': dict(sorted(Counter(f"{r.get('agent')}|{r.get('floor_status')}"
                                                              for r in records).items())),
                    'attempts': dict(sorted(Counter(r.get('attempt') for r in records).items()))}
    checks['S7_records_pass_per_record_predicate_at_production_tol'] = bool(records) and not bad
    checks['S8_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    pkl = os.path.join(eval_dir, 'certified_models.pkl')
    checks['S9_no_post_certification'] = rec.get('post_certification') is None and not os.path.exists(pkl)
    checks['S10_eval_key_equals_pair_x0_and_v34'] = (rec.get('eval_key') == entry['eval_key'] == ss['cells']['x0'][
        'eval_key'] == ss['cells']['x0']['v34_frozen_eval_key'])
    checks['S11_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    checks['S12_alpha_row_capture'], detail['S12'] = W9.alpha_row_capture_check(rec, eval_dir)
    checks['S13_sigma_inside_band'], detail['S13'] = W9.sigma_check(rec)
    checks['S15_b_in_force_as_declared'], detail['S15'] = b_plumbing_check(
        rec, records, STAGES[arm]['release_solution_bookkeeping'])
    return checks, detail, rec


def _cycle_rows_for_compare(eval_dir):
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    return [{k: v for k, v in r.items() if k not in NON_DETERMINISTIC_CYCLE_FIELDS} for r in rows]


def _record_for_compare(r):
    return {k: v for k, v in r.items() if k not in NON_DETERMINISTIC_RECORD_KEYS and not k.endswith('_path')
            and not k.endswith('_s')}


def bitwise_compare(dir_on, dir_off, rec_on, rec_off):
    """S16: per-cycle record rows identical on EVERY field except NON_DETERMINISTIC_CYCLE_FIELDS (JSON text, sort_keys:
    floats compared by their shortest repr, i.e. bit for bit); same row count == cap; the initialisation identity's
    gross hex equal. Reported, not gated: the per-solve IPOPT records compared on every non-path / non-time key."""
    a, b = _cycle_rows_for_compare(dir_on), _cycle_rows_for_compare(dir_off)
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
            diffs.append({'row': i, 'fields': sorted(k for k in set(x) | set(y)
                                                     if json.dumps(x.get(k)) != json.dumps(y.get(k)))})
    ii_on = (rec_on.get('initialisation_identity') or {}).get('gross_operational_cost_hex')
    ii_off = (rec_off.get('initialisation_identity') or {}).get('gross_operational_cost_hex')
    ra = _read_jsonl(os.path.join(dir_on, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    rb = _read_jsonl(os.path.join(dir_off, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    rec_diffs = [i for i, (x, y) in enumerate(zip(ra, rb))
                 if json.dumps(_record_for_compare(x), sort_keys=True) != json.dumps(_record_for_compare(y),
                                                                                    sort_keys=True)]
    diff_keys = sorted({k for i in rec_diffs[:50] for k in set(ra[i]) | set(rb[i])
                        if json.dumps(ra[i].get(k)) != json.dumps(rb[i].get(k))})
    parts = {'same_row_count_eq_cap': len(a) == len(b) == SMOKE_CAP, 'per_cycle_rows_identical': not diffs,
             'initialisation_identity_gross_hex_equal': bool(ii_on) and ii_on == ii_off,
             'cycles_run_equal': rec_on.get('cycles_run') == rec_off.get('cycles_run')}
    return all(parts.values()), {
        'parts': parts, 'n_rows': [len(a), len(b)], 'row_diffs': diffs[:10],
        'excluded_fields': list(NON_DETERMINISTIC_CYCLE_FIELDS), 'compared_fields': sorted(a[0]) if a else [],
        'gross_by_cycle_on': [r.get('gross_operational_cost') for r in a],
        'gross_by_cycle_off': [r.get('gross_operational_cost') for r in b],
        'init_identity_gross_hex': [ii_on, ii_off],
        'solve_records_reported_not_gated': {'n': [len(ra), len(rb)], 'n_differing': len(rec_diffs),
                                             'first_differing': rec_diffs[:10], 'differing_keys_first50': diff_keys,
                                             'excluded_keys': list(NON_DETERMINISTIC_RECORD_KEYS) + ['*_path', '*_s']}}


PERSISTENCE_MARGIN_RULE = {
    'rule': ('Addendum 48: persistence ONLY IF the measured runtime peak with persistence <= 0.85 x memory available '
             'after the reboot (>= 15 % headroom for a ~15 h run); the rule, not any prediction, decides'),
    'A_post': ('the available memory the smoke launch\'s preflight records BEFORE its first arm (L.memory_preflight: '
               'hw.memsize - (wired + anonymous + compressor-occupied) x page size, the vm_stat measure used by every '
               'committed launch) -- the smoke is the first launch after the author\'s reboot; recorded in '
               'smoke_gate.json provenance'),
    'P_on': ('the (b)-ON smoke arm\'s child peak, measured directly on the 3 x 3 instance: evaluation_record '
             'peak_rss.child_python_process_ru_maxrss (ru_maxrss of the evaluation process, bytes on macOS)'),
    'g': ('the measured growth from a smoke to a full certified run: max child peak of the 2 x 2 alpha-row pair 1 / the '
          '2 x 2 smoke r2 child peak (= k_run / k_smoke of the committed v34 memory model; both (b) off)'),
    'T_persist': 'the stage spec\'s declared persistence transient (ITEM 1: the max of every recorded 3 x 3 transient)',
    'P_run': 'g x P_on (the pair\'s per-child peak without persistence)',
    'P_persist': 'P_run + T_persist (the runtime peak with persistence; the transient on top of the peak: an upper bound)',
    'test': 'persist_certified_models = (P_persist <= 0.85 x A_post); hull_polish = False in every case',
    'where_decided': ('computed and recorded by the smoke gate (smoke_gate.json margin_rule), read by --stage pair '
                      '--freeze into every pair entry\'s post_certification; the pair --run preflight then REFUSES '
                      'unless available >= P_persist (persist) or P_run (no persist) at its own launch'),
    'without_persistence': ('the hull polish is omitted at 3 x 3 and reported as such, with the SRP1 figure (3.09e-6 '
                            'relative, below the band -- Addendum 48\'s statement; the terminal models are not kept); '
                            'with persistence the pickle is kept and no polish runs in the child either (Addendum 46 '
                            'voided the polished convention; a polish, if ever wanted, runs post hoc from the pickle)'),
}


def margin_rule(ss, peak_on_bytes, available_post_reboot_bytes):
    est = ss['item1_memory_measurement']['estimates']
    g = est['g_growth_cap_to_full_run']
    t = est['persist_transient']['T_persist_bytes']
    if not (isinstance(peak_on_bytes, int) and isinstance(available_post_reboot_bytes, int)):
        return {'evaluated': False, 'why': 'P_on or A_post missing', 'persist_certified_models': False,
                'P_on_bytes': peak_on_bytes, 'A_post_bytes': available_post_reboot_bytes}
    p_run = g * peak_on_bytes
    p_persist = p_run + t
    threshold = MARGIN * available_post_reboot_bytes
    persist = p_persist <= threshold
    return {'evaluated': True, 'definition': PERSISTENCE_MARGIN_RULE, 'P_on_bytes': peak_on_bytes, 'g': g,
            'T_persist_bytes': t, 'P_run_bytes': p_run, 'P_persist_bytes': p_persist,
            'A_post_bytes': available_post_reboot_bytes, 'threshold_bytes': threshold,
            'gib': {'P_on': peak_on_bytes / GIB, 'P_run': p_run / GIB, 'P_persist': p_persist / GIB,
                    'A_post': available_post_reboot_bytes / GIB, 'threshold': threshold / GIB},
            'persist_certified_models': persist,
            'post_certification_for_the_pair': dict(POST_CERT_PERSIST if persist else POST_CERT_NO_PERSIST),
            'required_pair_bytes': p_persist if persist else p_run}


# ======================================================================================================================
#  the stage spec v35
# ======================================================================================================================
SMOKE_GATE = {
    'applies_to': ('BOTH arms (bon: release_solution_bookkeeping True; boff: False) for S1-S13 and S15; S14, S16, S17 '
                   'are parent / cross-arm'),
    'S1': f'child exit 0; record written by the child; status not_certified; cycles_run == {SMOKE_CAP}',
    'S2': 'W86 S2: append byte-identical to the end-of-run file, sha256 == the record\'s',
    'S3': 'W86 S3: the tail checklist is line 1 of the events file, written before any IPOPT output',
    'S4': 'W86 S4: tail state check match (enabled, 1e-6)',
    'S5': (f'solve profile reconciled per event: base 83 x {SMOKE_CAP + 1} = {83 * (SMOKE_CAP + 1)}, observed == base + '
           'retries attempted, 0 blocked, network records == observed - 3 x rounds'),
    'S6': (f'the tail INACTIVE at cap {SMOKE_CAP}: no active / acted cycle; apply events at cycles 1..{SMOKE_CAP} then '
           f'the exit restore (None); next_state [False] x {SMOKE_CAP}; no cycle_convergence row; no "Convergence-depth '
           'tail ON" line'),
    'S7': 'every record passes the v32 per-record predicate with W empty (production compl_inf_tol)',
    'S8': 'W86 S8: last event sealed, no append write error',
    'S9': 'no post-certification requested: the record carries none and no certified_models.pkl exists',
    'S10': 'record eval_key == the smoke entry key == the pair x0 key == the v34 frozen x0 key',
    'S11': 'ESS ageing read-back all_match before and after',
    'S12': 'the alpha-row capture (v34 G10 / S12, W89 launcher alpha_row_capture_check)',
    'S13': 'sigma_computed / sigma_fixed inside production\'s band [1/3, 3]',
    'S14': 'every parent guard verify(0) == []',
    'S15': ('option (b) in force exactly as declared: the record declares it, the child\'s setter took effect with every '
            'network\'s read-back equal to the declaration, and the release call count == the number of successful '
            'network solves over all rounds (blocks whose final attempt is accepted) for bon, == 0 for boff'),
    'S16': ('BITWISE ((b) on vs off): per_cycle_record.jsonl rows identical on every field except '
            f'{list(NON_DETERMINISTIC_CYCLE_FIELDS)} -- per-cycle gross, recourse, objective change, every Boyd primal / '
            'dual residual ratio and channel pass, rho, the response record -- compared as JSON text (bit for bit); row '
            f'count == {SMOKE_CAP} on both; the initialisation identity\'s gross hex equal; cycles_run equal'),
    'S17': ('the 3 x 3 runtime peak MEASURED DIRECTLY on both arms: child_python_process_ru_maxrss and '
            'production_state_peak_rss_ru_maxrss present (ints) on both records'),
    'pass_iff': 'every S1-S17 check holds on every arm; the margin-rule verdict is recorded, NOT part of PASS',
}
SMOKE_REPORTED = ('per arm: child and production-state peaks, per-cycle RSS / ru_maxrss, cycle wall; the measured (b) '
                  'saving at runtime (child peak off - on, production peak off - on); floor-status tally, retries; the '
                  'persistence margin rule (A_post, P_on, g, T_persist, P_run, P_persist, threshold, verdict)')
PER_ENTRY_GATES = {
    **{k: v for k, v in W9.PER_ENTRY_GATES.items() if k not in ('G6_floor_records_v32', 'G8_post_certification')},
    'applies_to': 'both pair cells (x0, n7_4h_e1); there is no other arm',
    'G6_floor_records_v35': 'G6_V35 below (final scope; tier-2 ruling), B = 80',
    'G8_post_certification': ('as the pair campaign spec decided from the smoke gate\'s margin rule, resolved by '
                              'H.resolve_post_certification: nothing requested (no persist, no polish) -> no '
                              'post-certification in the record and no certified_models.pkl; persist -> certified: '
                              'status evaluated and the pickle written with sha256 == the record\'s; not certified: '
                              'skipped, no pickle'),
    'G12_b_in_force': 'as S15 for (b) on: declared True, took effect, read back True, calls == successful network solves',
    'reported_with_G6': ('per cell: judged applicable / not applicable (tier-2) counts in T; a tier-2 final attempt in '
                         'T is flagged for the Planner'),
}


def _instance_block(inst, derived):
    return {'record': {'path': W9.INSTANCE_RECORD_REL, 'sha256': _sha(W9.INSTANCE_RECORD_REL)},
            'case': {'path': W9.INSTANCE_CASE_REL, 'sha256': derived['case_sha256']},
            'derived_instance_declaration': derived, 'prefix_verification': inst['prefix_verification'],
            'prefix_draw': inst['prefix_draw'], 'facts': inst['facts'], 'checks': inst['checks']}


def predictions(est, inst, refs):
    s = est['saving_per_child']
    return {
        'recorded': ('BEFORE any 3 x 3 evaluation exists; blind to every 3 x 3 ADMM quantity (no 3 x 3 solve has ever '
                     'run); NOT blind to the instance facts, the W89 probes and the W90 zero-solve (b) probes, which '
                     'the Worker ran before writing these'),
        'R_restated': ('R = 0.9331 (W52 R_r2 of the prefix draw [1, 2, 3]) recorded; the ratio is value_3x3 / '
                       f"{refs['R_SRP1_tail']:.2f} (the SRP1 tight-tail reference, NOT the superseded "
                       f"{refs['R_SRP1_pre_tail_superseded']:.2f})"),
        'P1_ratio_band': W9.predictions(None, inst, refs)['P1_ratio_band'],
        'P1_resolution_caveat': ('at 2 x 2 the ratio resolution was 0.152: a 3 x 3 point inside [0.93, 1.09] means only '
                                 '"not distinguishable" from the band, nothing sharper; the 3 x 3 resolution is reported '
                                 'by the alpha-row formula'),
        'W1_worker_point': W9.predictions(None, inst, refs)['W1_worker_point'],
        'W2_value_minus_I_negative': W9.predictions(None, inst, refs)['W2_value_minus_I_negative'],
        'W3_certifies': W9.predictions(None, inst, refs)['W3_certifies'],
        'W5_retries': W9.predictions(None, inst, refs)['W5_retries'],
        'W6_rule_ten': W9.predictions(None, inst, refs)['W6_rule_ten'],
        'W7_sigma': W9.predictions(None, inst, refs)['W7_sigma'],
        'M1_b_saving_per_child': (f"{s['central_gib']:.2f} GiB (range {s['range_gib'][0]:.2f}-{s['range_gib'][1]:.2f}); "
                                  'Addendum 48 predicted ~3 GiB'),
        'M2_smoke_peaks': (f"(b) off arm {est['smoke_off_gib']:.2f} GiB (the v34 smoke model); (b) on arm "
                           f"{est['smoke_on_gib']['central_gib']:.2f} GiB (range {est['smoke_on_gib']['low_gib']:.2f}-"
                           f"{est['smoke_on_gib']['high_gib']:.2f}); the runtime saving (off - on) near M1"),
        'M3_pair_peak_no_persist': (f"(b) on {est['sustained_on_gib']['central_gib']:.2f} GiB (range "
                                    f"{est['sustained_on_gib']['low_gib']:.2f}-{est['sustained_on_gib']['high_gib']:.2f});"
                                    f" (b) off {est['sustained_off_gib']:.2f} GiB (v34)"),
        'M4_pair_peak_with_persist': (f"(b) on {est['persist_peak_on_gib']['central_gib']:.2f} GiB (high "
                                      f"{est['persist_peak_on_gib']['high_gib']:.2f}); (b) off "
                                      f"{est['persist_peak_off_v34_gib']:.2f} GiB (v34); T_persist "
                                      f"{est['persist_transient']['T_persist_gib']:.2f} GiB"),
        'M5_margin_rule': ('persistence will NOT be decided on (Addendum 48: ~24 GiB against ~24): '
                           f"P_persist ~ {est['persist_peak_on_gib']['central_gib']:.1f} GiB against 0.85 x A_post; it "
                           f"would need A_post >= {est['persist_peak_on_gib']['central_gib'] / MARGIN:.1f} GiB -- the rule, "
                           'not this prediction, decides'),
        'T1_cycle_time': ('10.1-12.5 min per cycle at concurrency 1 (the v34 estimate; the smoke measures it: 3 cycles '
                          'per arm)'),
        'T2_wall': ('smoke ~1.6-2.0 h for both arms (cap 3; Addendum 48 said ~1.2 h); per evaluation 12.6-15.7 h (range '
                    '10-20 h); the pair sequential 25-31 h'),
        'F1_floor_status': ('every terminal-round final accepted attempt applicable and "at" the mu floor with '
                            'compl_inf_tol 1e-6 in force on both cells (80 / 80); 0 tier-2 final attempts in T; the tail '
                            'active on the last ~9-12 cycles; pre-tail records reported, several "above" (the 2 x 2 x0 '
                            'TSO early stops were fifteen)'),
        'S1_smoke_bitwise': 'S16 holds: per-cycle gross and every residual ratio identical bit for bit on (b) on vs off',
    }


def stage_spec_content(inst, derived, probes, est, pre, rule11, plumbing, selftests, identity, v34):
    refs = W9._reference_R()
    cells = {}
    for label in STAGES['pair']['labels']:
        cells[label] = {'nodes': {str(n): list(v) for n, v in W9.CELLS[label].items()}, 'investment_year': W9.YEAR,
                        'candidate_key': W9._key_of(label), 'eval_key': W9._eval_key(label, derived),
                        'v34_frozen_eval_key': v34['cells'][label]['eval_key'],
                        'pre_tail_3x3_key': pre['per_cell'][label]['pre_tail_3x3_key'],
                        'alpha_row_2x2_key_same_candidate': pre['per_cell'][label]['alpha_row_2x2_key_same_candidate']}
    b_gate = _load(SRP1_B_GATE['path'])
    return {
        'schema': f'p515_frozen_spec_v{SPEC_VERSION}', 'version': SPEC_VERSION, 'stage': STAGE_TEXT,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 48 (all items for the 3 x 3 pair)',
                      'Planner task W90 (item 1: the (b) measurement at zero solves; item 2: spec v35; the Planner '
                      'ruling on W89 question 4)', 'Addenda 44 / 46 (prefix draw, R, tail) through v34'],
        'predecessor': {'path': SPEC_V34['path'], 'sha256': _sha(SPEC_V34['path'])},
        'predecessor_not_edited': 'v34 stays as frozen; its r2 campaign specs (never run) are superseded by v35',
        'predecessor_chain': {'v34': SPEC_V34, 'v33': W9.SPEC_V33, 'v32': W9.SPEC_V32},
        'superseded_campaign_freezes_under_v34': {
            'smoke': os.path.join(W9.ROOT_REL, 'campaign_s53_w89_3x3_smoke_r2'),
            'pair': os.path.join(W9.ROOT_REL, 'campaign_s53_w89_3x3_pair_r2'),
            'why': ('never run; they pin the pre-W90 harness sha (the (b) plumbing changed it), carry no (b) option, a '
                    'cap-2 one-arm smoke and v34\'s persistence decision')},
        'changes_from_v34': [
            'option (b) ON for the pair (entry option release_solution_bookkeeping True; harness plumbing added in W90; '
            'not keyed)', 'concurrency 1 fixed (Addendum 48 decision 1)',
            'persistence decided by the margin rule (operational test below), not by the v34 memory-model rule',
            'without persistence the hull polish is omitted at 3 x 3 and reported with the SRP1 figure',
            f'the smoke is TWO arms (bon / boff), x0, cap {SMOKE_CAP}, a bitwise comparison that measures the 3 x 3 '
            'runtime peak directly; distinct campaign ids -> distinct P56A/evals working dirs',
            'G6 v35: the final scope written for every future cell; the tier-2 non-vacuity ruling; the v30 / v32 '
            're-scopings recorded as post-hoc', 'identity_holds recomputed under the current formula (reference chain)',
            'G12 added ((b) in force)', 'predictions restated (memory with / without (b) and persistence)'],
        'instance': _instance_block(inst, derived),
        'instance_equals_v34': _instance_block(inst, derived) == v34['instance'],
        'prefix_draw_carried_over': {'market': W9.PREFIX_SUBSET, 'operation': W9.PREFIX_SUBSET,
                                     'verified_on_realized_arrays': inst['prefix_verification']['prefix_holds'],
                                     'R_recorded': W9.R_PREFIX_RECORDED},
        'configuration': {
            'name': (f'{W9.LABEL}; the 3 x 3 prefix-draw instance {W9.INSTANCE_LABEL}; row 18 premium alpha 0.5 (no '
                     'floor); case-file AA declared; the convergence-depth tail DECLARED {True, 1e-6}; option (b) '
                     'release_solution_bookkeeping ON; concurrency 1; persistence by the margin rule; no hull polish'),
            'arm_label': W9.ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': W9.CASE_FILE_AA,
            'ess_ageing_baseline': W9.ESS_AGEING_BASELINE, 'ess_ageing_baseline_label': W9.LABEL,
            'ess_params_file_sha256': W9.ESS_PARAMS_SHA256, 'derived_instance': derived,
            'convergence_depth_tail': W9.TAIL, 'interface_deviation_premium': W9.PREMIUM,
            'release_solution_bookkeeping': True, 'release_solution_bookkeeping_enters_evaluation_key': False,
            'post_certification': ('DECIDED BY THE MARGIN RULE at the smoke (persistence_margin_rule); '
                                   f'{POST_CERT_PERSIST} if it passes, else {POST_CERT_NO_PERSIST} (resolved to None)'),
            'hull_polish': False, 'cap': STAGES['pair']['cap'],
            'required_consecutive_cycles': W9.REQUIRED_CONSECUTIVE_CYCLES, 'concurrency': CONCURRENCY,
            'case_file_sha256': H.sha256_file(H.CASE_FILE),
            'per_cycle_response_capture': 'ON (harness: every derived-instance evaluation; asserted, rule eleven)',
            'floor_status_capture': 'ON (harness: every evaluation; asserted, rule eleven)'},
        'hull_polish_at_3x3': PERSISTENCE_MARGIN_RULE['without_persistence'],
        'persistence_margin_rule': PERSISTENCE_MARGIN_RULE,
        'cells': cells,
        'eval_key_carry_over': {label: {'v34': cells[label]['v34_frozen_eval_key'], 'v35': cells[label]['eval_key'],
                                        'equal': cells[label]['v34_frozen_eval_key'] == cells[label]['eval_key']}
                                for label in cells},
        'release_solution_bookkeeping_and_evaluation_key': {
            'enters_evaluation_key': plumbing['enters_evaluation_key'],
            'evidence': ('H.evaluation_key has no such parameter and its source never names it; three scratch campaign '
                         'freezes of x0 (option absent / True / False) give one identical eval key, equal to v34\'s '
                         '(harness_b_plumbing_checks K1)'),
            'consequence': 'the v34 frozen keys carry over (x0 f6e9cd53..., n7_4h_e1 c82522f4...)'},
        'harness_b_plumbing_checks': plumbing,
        'srp1_b_bitwise_gate_carried_over': {
            'path': SRP1_B_GATE['path'], 'sha256': _sha(SRP1_B_GATE['path']), **_git_state(SRP1_B_GATE['path']),
            'gate_pass': b_gate.get('gate_pass'), 'what': b_gate.get('stage'),
            'scope_note': ('SRP1, two cycles at C*, run through p515_s49_memory_fix_gate (the same setter the harness now '
                           'uses); the 3 x 3 smoke S16 re-tests bitwise identity on this instance under this harness')},
        'pre_launch_assertion': pre,
        'item1_memory_measurement': {
            'probes': {w: {'path': p['path'], 'sha256': p['sha256']} for w, p in probes.items()},
            'method': probes['s53_3x3_off']['data']['method'],
            'limitations': probes['s53_3x3_off']['data']['limitations'],
            'estimates': est},
        'declared_solve_profile': {
            'pair_per_cell': {'rule': ('83 x (cycles_run + 1) + every retry attempted (per event); cycles_run <= 500; '
                                       'no post-certification solve'),
                              'verification': 'RECONCILED PER EVENT in the child record (G5); NOT guard-verified'},
            'smoke_per_arm': {'rule': f'83 x ({SMOKE_CAP} + 1) = {83 * (SMOKE_CAP + 1)} + every retry attempted',
                              'base': 83 * (SMOKE_CAP + 1)},
            'parent': 'this launcher: SolveProfileGuard(permitted=()) and every imported launcher guard verify(0)'},
        'per_entry_gates': PER_ENTRY_GATES,
        'G6_V35': G6_V35, 'g6_post_hoc_record': G6_POST_HOC_RECORD,
        'g6_v35_self_tests_declared': G6_V35_SELF_TESTS, 'g6_v35_self_test_results': selftests,
        'g6_v32_verbatim': _load(W9.SPEC_V32['path'])['per_entry_gates']['G6_floor_records_v32'],
        'acceptance_cross_check': X.ACCEPTANCE_CROSS_CHECK,
        'smoke_stages': {k: STAGES[k] for k in SMOKE_ARMS}, 'smoke_gate_declared_before_run': SMOKE_GATE,
        'smoke_reported_not_gated': SMOKE_REPORTED,
        'smoke_required_before_pair': ('the pair --freeze and --run REFUSE unless the smoke gate is committed, clean and '
                                       'PASS'),
        'memory_preflights': {
            'smoke': ('before EACH arm (refusing): available >= the arm\'s estimate (bon: SMOKE_on high; boff: '
                      'SMOKE_off); the first measurement is A_post'),
            'pair': 'available >= margin_rule.required_pair_bytes at the pair launch (refusing)'},
        'solve_identity': identity,
        'objective_convention': W9.OBJECTIVE_CONVENTION,
        'value_definition': ('value = Q(x0) - Q(n7_4h_e1) on this instance (Q = certified_cost, gross, settlement '
                             'excluded); resolution = bar_x0 + bar_unit; |value| or |value - I| <= resolution is '
                             'INDETERMINATE (the bar bounds stopping slack only); ratio = value / 259,375.33; rule ten '
                             'reported for both cells'),
        'reference_R': refs, 'alpha_row_2x2_context': W9._alpha_row_2x2_context(),
        'predictions_recorded_before_any_3x3_run': predictions(est, inst, refs),
        'estimates_time': W9.estimates(W9.memory_model()[0], CONCURRENCY, False)['cycle_time'],
        'estimates_wall': {**W9.estimates(W9.memory_model()[0], CONCURRENCY, False)['wall_per_evaluation_h'],
                           'smoke_two_arms_h': [2 * ((SMOKE_CAP + 1) * 10.1 * 60 + 300) / 3600,
                                                2 * ((SMOKE_CAP + 1) * 12.5 * 60 + 600) / 3600]},
        'rule_eleven': rule11,
        'not_permitted': ['no change to the ADMM formulation, the certification criterion, the AA predicate, the tail or '
                          'any solver option (option (b) is the declared exception; it changes no model)',
                          'no committed artifact modified or re-run onto; fresh campaign ids / roots / working dirs',
                          'no screen / nohup / &; attached, alone, both streams captured'],
        'harness_sha256': H.sha256_file(H.HARNESS_PATH), 'launcher_sha256': H.sha256_file(os.path.abspath(__file__)),
        'w89_launcher_sha256': H.sha256_file(os.path.abspath(W9.__file__)),
        'w89_step1_sha256': H.sha256_file(os.path.abspath(X.__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def _find_stage_spec():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    sha = _sha(rel)
    if hits[0] != f'{SPEC_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'v{SPEC_VERSION} file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def load_stage_spec():
    rel, sha = _find_stage_spec()
    if rel is None:
        raise RuntimeError(f'frozen stage spec v{SPEC_VERSION} not found')
    return rel, sha, _load(rel)


def _common_checks():
    failures, ev = W9._common_checks()
    if _sha(SPEC_V34['path']) != SPEC_V34['sha256'] or not _committed_clean(SPEC_V34['path']):
        failures.append('v34 not as committed')
    if _sha(SRP1_B_GATE['path']) != SRP1_B_GATE['sha256'] or not _committed_clean(SRP1_B_GATE['path']):
        failures.append('the SRP1 (b) gate record not as committed')
    if _sha(W32_PROFILE) != W32_PROFILE_SHA256 or not _committed_clean(W32_PROFILE):
        failures.append('the W32 memory profile not as committed')
    for rel in (SCRIPT_NAME, 'p515_s53_w89_3x3_campaign.py', os.path.relpath(H.HARNESS_PATH, REPO)):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a 3 x 3 launcher is alive: {others}')
    return failures, ev


def freeze_spec(started):
    tag = f'W90-V{SPEC_VERSION}'
    failures, _ev = _common_checks()
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'v{SPEC_VERSION} already exists (write-once): {existing}')
    inst, derived = W9.load_instance_record()
    v34 = _load(SPEC_V34['path'])
    probes, pf = load_probes()
    failures += pf
    v34_model, err = W9.memory_model()
    if err:
        failures.append(err)
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    pre = pre_launch_assertion(derived)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails: {pre}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    plumbing = harness_b_plumbing_checks(derived)
    selftests, st_ok = g6_v35_self_tests()
    identity = solve_identity_recomputed()
    est = memory_estimates(probes, v34_model)
    val = est['srp1_validation']
    problems = []
    if not plumbing['all_hold']:
        problems.append(f"harness (b) plumbing checks fail: {[k for k, v in plumbing['checks'].items() if not v]}")
    if not st_ok:
        problems.append(f"G6 v35 self-tests not as declared: {[r['id'] for r in selftests.get('results', []) if not r['ok']]}")
    if not identity['all_recomputed_true']:
        problems.append('identity_holds does not recompute True on every reference-chain cell')
    if not (val['real_pair'] or {}).get('all_checks_pass'):
        problems.append(f"synthetic .sol validation against the real pair fails: {(val['real_pair'] or {}).get('checks')}")
    if not all(val['setter_check_took_effect'].values()):
        problems.append('set_release_solution_bookkeeping did not take effect on the SRP1 planning object')
    for w in ('s53_3x3_off', 's53_3x3_on'):
        t = probes[w]['data']['totals']
        if not (t['n_blocks'] == W9.BLOCKS_PER_ROUND and t['all_results_succeeded']):
            problems.append(f'b probe {w}: blocks {t["n_blocks"]} / results succeeded {t["all_results_succeeded"]}')
    if not probes['s53_3x3_on']['data']['totals']['after_release_all_empty']:
        problems.append('the ON probe did not leave every block\'s bookkeeping empty')
    if (probes['s53_3x3_on']['data'].get('pickle_step') or {}).get('status') != 'measured':
        problems.append('the ON probe did not measure the pickle transient')
    content = stage_spec_content(inst, derived, probes, est, pre, rule11, plumbing, selftests, identity, v34)
    if not content['instance_equals_v34']:
        problems.append('the instance block differs from v34\'s')
    if not all(v['equal'] for v in content['eval_key_carry_over'].values()):
        problems.append('an eval key does not carry over from v34')
    if content['srp1_b_bitwise_gate_carried_over']['gate_pass'] is not True:
        problems.append('the SRP1 (b) gate record does not read PASS')
    if problems:
        for p in problems:
            _log(f'[{tag} CHECK FAILED] {p}')
        _finish(1)
    m_now = L.memory_preflight(1)
    content['memory_at_freeze_non_gating'] = {**m_now, 'note': 'recorded only; A_post is measured by the smoke launch'}
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError(f'v{SPEC_VERSION} written bytes do not hash to the name')
    s = est['saving_per_child']
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] wrote {rel} sha256={sha} (predecessor {content['predecessor']})")
    for label, c in content['cells'].items():
        _log(f"[{tag}]   {label}: eval_key {c['eval_key']} (v34 {c['v34_frozen_eval_key'][:16]}; carried over "
             f"{c['eval_key'] == c['v34_frozen_eval_key']})")
    _log(f"[{tag}] (b) saving per child: central {s['central_gib']:.3f} GiB, range {s['range_gib']}; route A "
         f"{ {k: round(v, 3) for k, v in est['route_A_saving_gib'].items()} }; route B {est['route_B_gib']}; route W "
         f"{est['route_W_gib']}")
    _log(f"[{tag}] peaks: smoke off {est['smoke_off_gib']:.2f} / on {est['smoke_on_gib']}; pair on "
         f"{est['sustained_on_gib']}; with persist {est['persist_peak_on_gib']} (T {est['persist_transient']['T_persist_gib']:.2f})")
    _log(f"[{tag}] plumbing all_hold {plumbing['all_hold']} (enters key: {plumbing['enters_evaluation_key']}); G6 v35 "
         f"self-tests {st_ok}; identity recomputed all True {identity['all_recomputed_true']}; pre-launch {pre['holds']}")
    _finish(0, '-- next: --stage smoke --freeze')


# ======================================================================================================================
#  campaign freezes and runs
# ======================================================================================================================
def campaign_root(stage):
    return _abs(os.path.join(ROOT_REL, f"campaign_{STAGES[stage]['campaign_id']}"))


def _entries(stage, post_certification=None):
    out = []
    for label in STAGES[stage]['labels']:
        opts = {'investment_year': W9.YEAR, 'interface_deviation_premium': dict(W9.PREMIUM),
                'release_solution_bookkeeping': STAGES[stage]['release_solution_bookkeeping']}
        if post_certification is not None:
            opts['post_certification'] = dict(post_certification)
        out.append((label, W9._nodes(label), opts))
    return out


def _configuration(ss):
    return {'name': ss['configuration']['name'], 'arm_label': W9.ARM_LABEL, 'overrides': {},
            'case_file_anderson_acceleration': dict(W9.CASE_FILE_AA),
            'ess_ageing_baseline': copy.deepcopy(W9.ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': W9.LABEL,
            'derived_instance': ss['configuration']['derived_instance'], 'convergence_depth_tail': dict(W9.TAIL),
            'note': ('no overrides, no model variant, no flexibility-price variant; row 18 alpha 0.5 per entry; option '
                     '(b) per entry (not keyed); tail declared')}


def validate_spec(stage, spec, derived, ss_pin, ss, post_certification=None):
    cfg = spec['configuration']
    entries = spec['candidates']
    extra = spec.get('extra') or {}
    st = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == st['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == list(st['labels']),
        'cap': spec.get('cap') == st['cap'], 'concurrency': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == W9.REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label': cfg.get('arm_label') == W9.ARM_LABEL, 'no_overrides': cfg.get('overrides') == {},
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == W9.CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == W9.ESS_AGEING_BASELINE,
        'ess_params_file_pinned': (cfg.get('ess_params_file') or {}).get('sha256') == W9.ESS_PARAMS_SHA256,
        'derived_instance_declared': cfg.get('derived_instance') == derived,
        'tail_declared_enabled_1e-6': cfg.get('convergence_depth_tail') == W9.TAIL,
        'no_model_variant': 'model_variant_label' not in spec and not any('model_variant' in e for e in entries),
        'no_flex_price_variant': 'flex_price_label' not in spec and not any('flex_price_multiplier' in e for e in entries),
        'stage_spec_pinned': extra.get('stage_spec') == ss_pin,
        'script_recorded': extra.get('campaign_script') == SCRIPT_NAME,
    }
    for e in entries:
        label = e['label']
        checks[f'{label}:canonical_key'] = e.get('key') == W9._key_of(label)
        checks[f'{label}:eval_key_stage_spec'] = (e.get('eval_key') == ss['cells'][label]['eval_key']
                                                  == W9._eval_key(label, derived)
                                                  == ss['cells'][label]['v34_frozen_eval_key'])
        checks[f'{label}:premium'] = e.get('interface_deviation_premium') == W9.PREMIUM
        checks[f'{label}:no_overrides'] = e.get('overrides') == {}
        checks[f'{label}:release_solution_bookkeeping'] = (e.get('release_solution_bookkeeping')
                                                           is st['release_solution_bookkeeping'])
        checks[f'{label}:post_certification'] = e.get('post_certification') == H.resolve_post_certification(
            post_certification, e['key'])
    return checks


def _launcher_pins_ok(ss):
    return (ss['launcher_sha256'] == H.sha256_file(os.path.abspath(__file__))
            and ss['harness_sha256'] == H.sha256_file(H.HARNESS_PATH))


def _freeze_one(stage, ss_rel, ss_sha, ss, derived, rule11, pre, post_certification=None, extra_more=None):
    st = STAGES[stage]
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'stage': stage, 'stage_text': STAGE_TEXT, 'label': W9.LABEL, 'stage_spec': ss_pin,
             'expected_eval_keys': {label: ss['cells'][label]['eval_key'] for label in st['labels']},
             'release_solution_bookkeeping': st['release_solution_bookkeeping'],
             'pre_launch_assertion_at_freeze': pre, 'objective_convention': W9.OBJECTIVE_CONVENTION,
             'solve_claim': ('RECONCILED PER EVENT, NOT GUARD-VERIFIED: the child record solve_profile (83 x (cycles_run '
                             '+ 1) + retries attempted); this parent\'s permitted=() guards verify(0)'),
             'rule_eleven': rule11, **(extra_more or {})}
    return H.freeze_campaign_spec(
        campaign_root(stage), st['campaign_id'], _entries(stage, post_certification), configuration=_configuration(ss),
        cap=st['cap'], concurrency=CONCURRENCY,
        authority=['PLANNER_BRIEF_2026-09-13.md Addendum 48', 'Planner task W90', ss_rel],
        required_consecutive_cycles=W9.REQUIRED_CONSECUTIVE_CYCLES, extra=extra), ss_pin


def _smoke_gate_rel():
    return os.path.join(SMOKE_GATE_DIR_REL, SMOKE_GATE_FILE)


def freeze(stage, started):
    tag = f'W90-{stage.upper()}-FREEZE'
    arms = SMOKE_ARMS if stage == 'smoke' else ('pair',)
    failures = []
    for arm in arms:
        failures += H.check_campaign_preconditions(campaign_root(arm), extra_clean_files=EXTRA_CLEAN_FILES)
    more, _ev = _common_checks()
    failures += more
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        if not _committed_clean(ss_rel):
            failures.append('the stage spec is not committed / clean')
        if not _launcher_pins_ok(ss):
            failures.append('the launcher or the harness changed since the stage spec froze')
    except RuntimeError as error:
        failures.append(str(error))
        ss = None
    inst, derived = W9.load_instance_record()
    if ss is not None and ss['configuration']['derived_instance'] != derived:
        failures.append('the instance record declaration differs from the stage spec')
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    pre = pre_launch_assertion(derived)
    if not pre['holds']:
        failures.append(f'pre-launch assertion (recomputed) fails: {pre}')
    post_cert, extra_more = None, None
    if stage == 'pair':
        rel = _smoke_gate_rel()
        gate = _load(rel) if os.path.isfile(_abs(rel)) else {}
        if not (_committed_clean(rel) and gate.get('pass') is True):
            failures.append(f"the smoke gate must be committed, clean and PASS: {rel} pass={gate.get('pass')}")
        mr = gate.get('margin_rule') or {}
        if not mr.get('evaluated'):
            failures.append('the smoke gate carries no evaluated margin rule')
        post_cert = mr.get('post_certification_for_the_pair')
        extra_more = {'smoke_gate': {'path': rel, 'sha256': _sha(rel) if os.path.isfile(_abs(rel)) else None},
                      'margin_rule_from_smoke_gate': mr, 'post_certification_decided': post_cert,
                      'required_pair_bytes': mr.get('required_pair_bytes')}
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    ok, specs = True, []
    for arm in arms:
        extra_arm = dict(extra_more or {})
        if stage == 'smoke':
            extra_arm['smoke_gate_declared_before_run'] = SMOKE_GATE
            extra_arm['declared_solves'] = {'base': 83 * (SMOKE_CAP + 1),
                                            'rule': f'83 per round x (cap {SMOKE_CAP} + 1) + every retry attempted'}
        (spec_path, spec_sha, spec), ss_pin = _freeze_one(arm, ss_rel, ss_sha, ss, derived, rule11, pre,
                                                          post_certification=post_cert, extra_more=extra_arm)
        checks = validate_spec(arm, spec, derived, ss_pin, ss, post_certification=post_cert)
        specs.append(spec)
        _log(f'[{tag}] {arm}: frozen campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
        for e in spec['candidates']:
            _log(f"[{tag}]   {e['label']}: eval_key={e['eval_key']} eval_dir={e['eval_dir']} working_dir_ids="
                 f"{e['working_dir_ids']} release_solution_bookkeeping={e.get('release_solution_bookkeeping')} "
                 f"post_certification={e['post_certification']}")
        _log(f'[{tag}] {arm}: spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
        ok = ok and all(checks.values())
    pre_frozen = pre_launch_assertion(derived, specs)
    _log(f"[{tag}] pre-launch assertion on the FROZEN specs: holds={pre_frozen['holds']} (committed keys scanned "
         f"{pre_frozen['n_committed_keys_scanned']})")
    if stage == 'smoke':
        ids = [e['working_dir_ids']['run'] for s in specs for e in s['candidates']]
        distinct = len(set(ids)) == len(ids)
        _log(f'[{tag}] working-dir ids distinct across arms: {distinct} {ids}')
        ok = ok and distinct
        m = L.memory_preflight(1)
        _log(f"[{tag}] memory at freeze (non-gating): available {m.get('available_gib')} GiB; required at run: bon "
             f"{ss['item1_memory_measurement']['estimates']['smoke_on_gib']['high_gib']:.2f} GiB, boff "
             f"{ss['item1_memory_measurement']['estimates']['smoke_off_gib']:.2f} GiB")
    ok = ok and pre_frozen['holds']
    _finish(0 if ok else 1, f'freeze {"OK" if ok else "NOT OK"}')


def _run_preconditions(stage, spec_sha256, ss_rel, ss_sha, ss, derived):
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append(f'campaign spec not committed / clean: {spec_path}')
    post_cert = (spec.get('extra') or {}).get('post_certification_decided') if stage == 'pair' else None
    checks = validate_spec(stage, spec, derived, {'path': ss_rel, 'sha256': ss_sha}, ss, post_certification=post_cert)
    failures += [f'{stage}: spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__))),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{stage}: {what} sha256 differs from the frozen spec')
    for e in spec['candidates']:
        for eid in e['working_dir_ids'].values():
            if os.path.exists(os.path.join(L._work_dir(), eid)):
                failures.append(f'working dir already exists (never reusable): {eid}')
    return root, spec_path, spec, failures


def _arm_preflight(required_bytes, stage):
    m = L.memory_preflight(1)
    m = {**m, 'stage': stage, 'required_bytes': required_bytes, 'required_gib': required_bytes / GIB,
         'w86_rule_fields_superseded': ['required_bytes', 'required_gib', 'rule', 'per_child_budget_gib']}
    m['pass'] = m.get('available_bytes') is not None and m['available_bytes'] >= required_bytes
    return m


def _manifest_of(paths):
    return W9._manifest_of(paths)


def run_smoke(started, sha_on, sha_off):
    tag = 'W90-SMOKE'
    failures, _ev = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not (_committed_clean(ss_rel) and _launcher_pins_ok(ss)):
        failures.append('stage spec not committed, or the launcher / harness changed since it froze')
    inst, derived = W9.load_instance_record()
    gate_dir = _abs(SMOKE_GATE_DIR_REL)
    if os.path.exists(gate_dir):
        failures.append(f'smoke gate dir exists (write-once): {gate_dir}')
    arms = {}
    for arm, sha in (('smoke_bon', sha_on), ('smoke_boff', sha_off)):
        root, spec_path, spec, f = _run_preconditions(arm, sha, ss_rel, ss_sha, ss, derived)
        failures += f
        arms[arm] = {'root': root, 'spec_path': spec_path, 'spec_sha256': sha, 'spec': spec}
    try:
        launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
    pre = pre_launch_assertion(derived, [a['spec'] for a in arms.values()])
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen specs: {pre}')
    est = ss['item1_memory_measurement']['estimates']
    required = {'smoke_bon': est['smoke_on']['high_bytes'], 'smoke_boff': est['v34_model_inputs']['smoke_bytes']}
    a_post = _arm_preflight(max(required.values()), 'smoke launch (A_post)')
    _log(f"[{tag}] A_post (post-reboot availability, recorded): {a_post.get('available_gib')} GiB")
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    per_arm = {}
    for arm in SMOKE_ARMS:
        a = arms[arm]
        entry = a['spec']['candidates'][0]
        eval_dir = os.path.join(a['root'], 'evals', entry['eval_dir'])
        pf = _arm_preflight(required[arm], arm)
        _log(f"[{tag}] {arm}: preflight available {pf.get('available_gib')} GiB required {pf['required_gib']:.2f} GiB -> "
             f"{'PASS' if pf['pass'] else 'REFUSE'}")
        batch, err = None, None
        if pf['pass']:
            lock = H.acquire_campaign_lock(a['spec']['campaign_id'], a['spec_sha256'])
            _log(f"[{tag}] {arm}: ONE cell {entry['label']} cap {a['spec']['cap']} release_solution_bookkeeping "
                 f"{entry.get('release_solution_bookkeeping')}; lock {lock}")
            try:
                ctx = H.CampaignContext(a['root'], a['spec_path'], a['spec_sha256'], a['spec'], log=_log)
                H.evaluate.last_batch_info = {}
                H.evaluate([entry['label']], ctx)
                batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
            except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
                err = f'{type(error).__name__}: {error}'
                print(traceback.format_exc(), file=sys.stderr, flush=True)
            finally:
                H.release_campaign_lock(expected_pid=os.getpid())
        try:
            checks, detail, rec = smoke_arm_checks(arm, entry, eval_dir, ss)
        except Exception as error:  # noqa: BLE001 -- the gate FAILS, recorded
            checks, detail, rec = ({'smoke_checks_ran': False},
                                   {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}, {})
            print(detail['traceback'], file=sys.stderr, flush=True)
        if not pf['pass']:
            checks['arm_preflight_passed'] = False
        if err:
            checks['evaluate_raised_nothing'] = False
        rows = (_read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
                if os.path.isfile(os.path.join(eval_dir, 'per_cycle_record.jsonl')) else [])
        per_arm[arm] = {'eval_dir': os.path.relpath(eval_dir, REPO), 'campaign_spec_path': os.path.relpath(
            a['spec_path'], REPO), 'campaign_spec_sha256': a['spec_sha256'], 'preflight': pf, 'evaluate_error': err,
                        'batch_info': batch, 'checks': checks, 'detail': detail, 'record': rec,
                        'peak_rss': rec.get('peak_rss'), 'wall_time_s': rec.get('wall_time_s'),
                        'cycle_wall_s': [r.get('cycle_wall_s') for r in rows],
                        'cycle_rss_bytes': [r.get('rss_bytes') for r in rows],
                        'cycle_ru_maxrss_bytes': [r.get('ru_maxrss_bytes') for r in rows]}
        for k, v in checks.items():
            _log(f'[{tag}]   {arm} {k}: {"PASS" if v else "FAIL"}')
    on, off = per_arm['smoke_bon'], per_arm['smoke_boff']
    cross = {}
    try:
        cross['S16_bitwise_on_vs_off'], cross_detail = bitwise_compare(_abs(on['eval_dir']), _abs(off['eval_dir']),
                                                                        on['record'], off['record'])
    except Exception as error:  # noqa: BLE001
        cross['S16_bitwise_on_vs_off'] = False
        cross_detail = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    peaks = {arm: {k: (per_arm[arm]['peak_rss'] or {}).get(k) for k in (
        'child_python_process_ru_maxrss', 'production_state_peak_rss_ru_maxrss')} for arm in SMOKE_ARMS}
    cross['S17_runtime_peak_measured_both_arms'] = all(isinstance(v, int) for p in peaks.values() for v in p.values())
    g = guards_verify()
    cross['S14_parent_guards_zero'] = _guards_ok(g)
    mr = margin_rule(ss, peaks['smoke_bon']['child_python_process_ru_maxrss'], a_post.get('available_bytes'))
    saving_runtime = {k: ((peaks['smoke_boff'][k] - peaks['smoke_bon'][k])
                          if all(isinstance(peaks[a][k], int) for a in SMOKE_ARMS) else None)
                      for k in ('child_python_process_ru_maxrss', 'production_state_peak_rss_ru_maxrss')}
    all_checks = {f'{arm}:{k}': v for arm in SMOKE_ARMS for k, v in per_arm[arm]['checks'].items()}
    all_checks.update(cross)
    for arm in SMOKE_ARMS:
        per_arm[arm].pop('record', None)
    gate = {'stage': STAGE_TEXT, 'gate': f'W90 two-arm 3 x 3 smoke (x0, cap {SMOKE_CAP}, (b) on vs off)', 'utc': _utc(),
            'git_head': H._git(['rev-parse', 'HEAD']), 'stage_spec': {'path': ss_rel, 'sha256': ss_sha},
            'A_post_memory_preflight_at_launch': a_post, 'per_arm': per_arm, 'cross_arm': cross,
            'cross_arm_detail': cross_detail, 'checks': all_checks, 'pass': all(all_checks.values()),
            'failing': sorted(k for k, v in all_checks.items() if not v),
            'runtime_peaks_measured_bytes': peaks, 'b_saving_measured_at_runtime_bytes': saving_runtime,
            'b_saving_measured_at_runtime_gib': {k: (v / GIB if v is not None else None) for k, v in
                                                 saving_runtime.items()},
            'margin_rule': mr, 'pre_launch_assertion': pre, 'guards': g, 'launcher_wall_s': time.time() - started}
    os.makedirs(gate_dir)
    H._write_once_json(os.path.join(gate_dir, SMOKE_GATE_FILE), gate)
    H._write_once_json(os.path.join(gate_dir, SMOKE_MANIFEST_FILE),
                       _manifest_of([arms[a]['root'] for a in SMOKE_ARMS] + [os.path.join(gate_dir, SMOKE_GATE_FILE)]))
    _log(f"[{tag}] runtime peaks {peaks}; (b) saving at runtime {gate['b_saving_measured_at_runtime_gib']}")
    _log(f"[{tag}] margin rule: {mr.get('gib')} -> persist {mr.get('persist_certified_models')}")
    _finish(0 if gate['pass'] else 1, f"GATE {'PASS' if gate['pass'] else 'FAIL'} failing={gate['failing']}")


def post_certification_check(rec, eval_dir, request):
    pc = rec.get('post_certification') or {}
    pkl = os.path.join(eval_dir, 'certified_models.pkl')
    resolved = H.resolve_post_certification(request, rec.get('candidate_key'))
    if resolved is None:
        ok = rec.get('post_certification') is None and not os.path.exists(pkl)
    elif rec.get('status') == 'certified':
        ok = (pc.get('status') == 'evaluated' and os.path.isfile(pkl)
              and (pc.get('persisted_models') or {}).get('sha256') == H.sha256_file(pkl))
    else:
        ok = pc.get('status') == 'skipped' and not os.path.exists(pkl)
    return ok, {'post_certification': pc, 'requested': request, 'resolved': resolved,
                'pickle': ({'bytes': os.path.getsize(pkl), 'sha256': H.sha256_file(pkl)} if os.path.isfile(pkl) else None)}


def cell_gates(entry, eval_dir, ss, request):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec:
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == ss['cells'][entry['label']]['eval_key'] == entry['eval_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d, 'note': 'SRP1-specific items superseded by G5 / G6'}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W9._solve_profile_check(rec, len(records))
    g6 = g6_v35_evaluate(eval_dir, rec, W9.BLOCKS_PER_ROUND)
    gates['G6_floor_records_v35'] = g6['gate_pass']
    detail['G6'] = g6
    detail['G6_tier2_final_attempts_in_T_flag'] = g6['tier2_terminal_blocks_counted']['n_judged_not_applicable'] > 0
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_post_certification'], detail['G8'] = post_certification_check(rec, eval_dir, request)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    gates['G10_alpha_row_capture'], detail['G10'] = W9.alpha_row_capture_check(rec, eval_dir,
                                                                               require_compared=entry['label'] == 'x0')
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G12_b_in_force'], detail['G12'] = b_plumbing_check(rec, records, True)
    return gates, detail, rec


def run_pair(started, spec_sha256):
    tag = 'W90-PAIR'
    failures, _ev = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not (_committed_clean(ss_rel) and _launcher_pins_ok(ss)):
        failures.append('stage spec not committed, or the launcher / harness changed since it froze')
    inst, derived = W9.load_instance_record()
    root, spec_path, spec, f = _run_preconditions('pair', spec_sha256, ss_rel, ss_sha, ss, derived)
    failures += f
    smoke_rel = _smoke_gate_rel()
    smoke = _load(smoke_rel) if os.path.isfile(_abs(smoke_rel)) else {}
    if not (_committed_clean(smoke_rel) and smoke.get('pass') is True):
        failures.append(f"the smoke gate must be committed, clean and PASS: pass={smoke.get('pass')}")
    extra = spec.get('extra') or {}
    if (extra.get('smoke_gate') or {}).get('sha256') != (_sha(smoke_rel) if os.path.isfile(_abs(smoke_rel)) else None):
        failures.append('the smoke gate changed since the pair spec froze')
    request = extra.get('post_certification_decided')
    try:
        launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
    pre = pre_launch_assertion(derived, [spec])
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre}')
    mem = _arm_preflight(int(extra.get('required_pair_bytes') or 0), 'pair launch')
    _log(f"[{tag}] memory preflight: available {mem.get('available_gib')} GiB required {mem['required_gib']:.2f} GiB -> "
         f"{'PASS' if mem['pass'] else 'REFUSE'}")
    if not extra.get('required_pair_bytes') or not mem['pass']:
        failures.append(f"memory preflight REFUSED: {mem.get('available_gib')} < {mem['required_gib']}")
    if failures:
        for fl in failures:
            _log(f'[{tag} PRECONDITION FAILED] {fl}')
        _finish(1)
    labels = list(STAGES['pair']['labels'])
    on_eval = smoke['per_arm']['smoke_bon']['eval_dir']
    src = os.path.join(campaign_root('smoke_bon'), H.INIT_IDENTITY_DIR_NAME, f'{os.path.basename(on_eval)}.json')
    dst_dir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    os.makedirs(dst_dir)
    shutil.copyfile(src, os.path.join(dst_dir, f'smoke_reference__{os.path.basename(src)}'))
    _log(f'[{tag}] placed the smoke (b)-on x0 initialisation record as the x0 identity reference ({H.sha256_file(src)})')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cells {labels} at concurrency {spec['concurrency']}; post_certification {request}; "
         f"lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    entries = {e['label']: e for e in spec['candidates']}
    per_cell, cells_q = {}, {}
    for label in labels:
        eval_dir = os.path.join(root, 'evals', entries[label]['eval_dir'])
        try:
            gates, detail, rec = cell_gates(entries[label], eval_dir, ss, request)
        except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
            gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                             'traceback': traceback.format_exc()}, {}
        cell = None
        if rec.get('status') in ('certified', 'not_certified'):
            try:
                cell = A.cell_quantities(os.path.relpath(eval_dir, REPO))
            except Exception as error:  # noqa: BLE001
                cell = {'error': f'{type(error).__name__}: {error}'}
        cells_q[label] = cell or {}
        per_cell[label] = {'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                           'certification_cycle': rec.get('certification_cycle'), 'Q': rec.get('certified_cost'),
                           'bar': (rec.get('bar') or {}).get('value'), 'peak_rss': rec.get('peak_rss'),
                           'gates': gates, 'gates_pass': all(gates.values()), 'gate_detail': detail,
                           'cell_quantities': cell, 'parent_view': (by_label.get(label) or {}).get('parent_view')}
    refs = ss['reference_R']
    value = W9.value_block(cells_q, refs, inst)
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'pre_launch_assertion': pre,
               'objective_convention': W9.OBJECTIVE_CONVENTION, 'per_cell': per_cell,
               'all_gates_pass': all(per_cell[k]['gates_pass'] for k in labels), 'value_and_R': value,
               'reference_R': refs, 'smoke_gate': {'path': smoke_rel, 'sha256': _sha(smoke_rel)},
               'post_certification_decided': request,
               'hull_polish_at_3x3': ss['hull_polish_at_3x3'],
               'solve_claim': 'RECONCILED PER EVENT in each child record (G5); parent guards verify(0)',
               'guards': g, 'memory_preflight_at_run': mem, 'batch_info': batch, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, PAIR_RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, PAIR_MANIFEST_FILE), _manifest_of([root]))
    for label in labels:
        p = per_cell[label]
        _log(f"[{tag}] {label}: status {p['status']} cycles {p['cycles_run']} Q {p['Q']} bar {p['bar']} gates {p['gates']}")
    _log(f'[{tag}] value / R: {value}')
    code = 0
    if not results['all_gates_pass'] or not _guards_ok(g):
        code = 1
    elif any(per_cell[k]['status'] != 'certified' for k in labels):
        code = 2
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--b-probe', choices=B_PROBES)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--stage', choices=('smoke', 'pair'))
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--spec-sha256-bon', default=None)
    parser.add_argument('--spec-sha256-boff', default=None)
    parser.add_argument('--scratch', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.b_probe:
            if not args.scratch or os.path.abspath(args.scratch).startswith(REPO + os.sep):
                parser.error('--scratch <dir outside the repository> is required')
            if not _committed_clean(SCRIPT_NAME) or not _committed_clean(os.path.relpath(H.HARNESS_PATH, REPO)):
                parser.error('the launcher and the harness must be committed and clean before a probe writes evidence')
            os.makedirs(args.scratch, exist_ok=True)
            b_probe(args.b_probe, started, args.scratch)
        elif args.freeze_spec:
            freeze_spec(started)
        else:
            if args.freeze == args.run:
                parser.error('--stage needs exactly one of --freeze / --run')
            if args.freeze:
                freeze(args.stage, started)
            elif args.stage == 'smoke':
                if not (args.spec_sha256_bon and args.spec_sha256_boff):
                    parser.error('--stage smoke --run requires --spec-sha256-bon and --spec-sha256-boff')
                run_smoke(started, args.spec_sha256_bon, args.spec_sha256_boff)
            else:
                if not args.spec_sha256:
                    parser.error('--stage pair --run requires --spec-sha256')
                run_pair(started, args.spec_sha256)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
