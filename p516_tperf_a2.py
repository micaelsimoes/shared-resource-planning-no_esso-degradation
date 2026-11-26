"""P5.16 T-perf A2 -- the warm-start barrier A/B (`mu_init`) on the certified SRP1 unit-cell blocks.

Authority: the author's "Decision 4" (2026-10-31; TASKS.md line "Expert R7 ... diagnostic A/B (<= 6 solves)"),
P5_16_TPERF_SPEC_v1.md (the T-perf diagnostic; not a production change; nothing here enters a table).
Baseline: P5_16_TPERF_REPORT.md (Part A, evidence data/SRP1/Results/P516TPERF/local, manifest fbfd6310...):
the warm re-solve took 22 (TSO) / 51 (DSO) iterations with the barrier restarting at mu = 0.1 (`mu_init` not passed).

INSTANCE: as Part A (imported from p516_tperf_profile: campaign s53_w86_tail_recert, cell n7_4h_e1, eval key
ca8927e75d628bd1..., certified at cycle 112), certified_models.pkl sha256 fe06ca48...efda04 verified against the
declared value and both committed manifests BEFORE unpickling; blocks TSO 2025 Spring (case9) and DSO node 7 2025
Spring (case33_2); every case solves its OWN clone; the pickle and the loaded blocks are never modified.

SOLVE PATH, TIMING, OPTION ASSEMBLY, LOGS: reused BY IMPORT from p516_tperf_profile.py (not modified): `run_case`
(network.run_smopf on the block's own build-time network/params, Instrument wrappers for NL write / IPOPT process /
sol read / load_from, per-case cap that blocks a production recovery from adding a solve), `parse_ipopt_log`,
`capture_checklist`, `Lock`, `verify_inputs`, `_write_json` (shared gate_result_io writer). Options per case: the
block's pickled production options + `output_file` (fresh per solve, asserted absent) + `print_timing_statistics yes`
(output only), + `mu_init` = MU_INIT on the mu cases. Nothing else changes (warm_start_init_point yes from
production's warm block, the tight bound/slack pushes, tol, MA97, file_print_level 6).

THE SIX DECLARED SOLVES (bounded SolveProfileGuard, permitted network.py:_run_smopf_solver_attempt, verify(6) exact):
  per block (TSO then DSO):
  c1_ident_mu  identical problem, warm start exactly as production, + mu_init
  c2_pert      perturbed problem (below), production options (no mu_init)
  c3_pert_mu   the same perturbed problem + mu_init
PERTURBATION ("perturbation = 10 x terminal mismatch"): every consensus-target Param the block carries -- vmag_req,
p_pf_req, q_pf_req (interface V / P / Q targets) and p_ess_req, q_ess_req (storage P / Q targets) -- is shifted by
+PERTURB_FACTOR x the block's recorded terminal inter-copy mismatch of that channel: the max |mismatch| over the
block's 2025 Spring entries in the T0.1b evidence data/SRP1/Results/P516D0/t01b/coupling.json (read-only; hash
verified against the committed T0.1b manifest): TSO -> nodes 5, 7, 9, channels v (from v_W: |sqrt(a) - sqrt(b)| kV),
pf_p (MW), pf_q (Mvar), ess_p_tso_esso (MW), ess_q_tso_esso (Mvar); DSO node 7 -> node 7, same interface channels,
ess_p_dso_esso, ess_q_dso_esso. Converted to the Param units exactly as production sets them (V / v_base kV, P / s_base).
Duals (dual_*), rho, prox_*_prev and everything else unchanged (asserted on the mutable-state signature).

GATES (Planner ruling (b), 2026-10-31; options = the pickled pre-tail production options for all six solves):
  c1   objective (Pyomo value at the loaded point) vs PART A'S WARM LIMIT under the same options -- the final
       unscaled objective of Part A's warm (i) IPOPT log, 17 digits (TSO -52244.700129902631, DSO 54561.135089701755)
       -- |f_c1 - f_A| / |f_A| <= 1e-9. c1's free primal Vars are REPORTED against the certified values (max abs
       deviation, p.u., per component), NOT gated: Part A stored no primal values and the certified point was produced
       under the tail options (compl_inf_tol 1e-6), a different limit.
  c3 vs c2 (same options, same perturbed problem): objective <= 1e-9 relative AND every free primal Var <= 1e-6.
Stop rules: a solve fails or the per-case cap blocks one, any restoration, an input hash mismatch, or the c1 objective
gate fails -> record and stop (no retries beyond production policy).

PRE-LAUNCH COMPARATOR CHECK (zero solves; fail-fast; a real run REFUSES, a dry run records and continues):
(i) the c1 comparator parsed from Part A's committed warm log (hash verified against Part A's manifest) equals the
ruling's declared 17-digit value exactly; (ii) the options c1 passes, less output_file and mu_init, equal Part A's
warm (i) options less output_file (same option regime). Recorded, not gated: the certifying solve's options from the
cell's production records and the certified-vs-Part-A-limit offset (finding carried from the stopped attempt).

OUTPUT (write-once; refuses if it exists): data/SRP1/Results/P516TPERF/a2/ -- a2.json, predictions.json (written
before the first solve), run.log, logs/ (IPOPT output_file + stdout/stderr per solve), nl/ (the NL files read),
manifest_sha256.json (--manifest, after the run; it also hashes the launch logs, both harnesses and the inputs).
LAUNCH (attached, alone, both streams captured), then the manifest:
    set -o noclobber && /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py \\
        > data/SRP1/Results/P516TPERF/a2_launch_stdout.log 2> data/SRP1/Results/P516TPERF/a2_launch_stderr.log
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py --manifest
Zero-solve dry run (blocking guard; run_smopf is reached and blocked; writes only under the new scratch directory and
the shared T-perf lock file):
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py --dry-run /tmp/<new dir>

CONTINUATION A2b (`--continue-from a2`; Planner ruling (a), 2026-10-31). The six declared solves span TWO launches:
a2 (1 solve: tso_c1_ident_mu, done; a2/manifest_sha256.json c22e94d3..., a2/a2.json e87d67b2...) + a2b (the five
unrun: tso c2, c3; dso c1, c2, c3). a2b's bounded guard permits exactly 5, verify(5) exact; TSO c1 is NOT re-run:
its record is read by reference (a2 manifest, a2.json and c1's IPOPT log hash-verified before anything is unpickled).
Gates restated to the stopping-slack basis (ruling (a)): two solves agree when |df| <= the max over the two solves of
their terminal UNSCALED complementarity (IPOPT log final statistics); ratio |df| / bar reported; ratio < 1 =
"indeterminate (inside stopping slack)" = pass; >= 1 = a real difference (recorded and flagged; the run is not
stopped on a gate). Applied to dso c1 vs Part A's DSO warm limit, c3 vs c2 on both blocks, and (zero solves, from
a2's record) tso c1 vs Part A's TSO warm limit. The 1e-9 relative figure is reported beside, not gated. c3 vs c2
Vars: max |dx| (p.u.) and the count above 1e-6 reported; agreement declared iff max |dx| <= 1e-4 p.u.; not a stop.
Stops: failed solve / per-case cap block, restoration, input or reference hash mismatch. a2 mode (no flag) is
unchanged. Output data/SRP1/Results/P516TPERF/a2b/ (write-once; refuses if it exists).
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py --continue-from a2 --dry-run /tmp/<new dir>
    set -o noclobber && /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py --continue-from a2 \\
        > data/SRP1/Results/P516TPERF/a2b_launch_stdout.log 2> data/SRP1/Results/P516TPERF/a2b_launch_stderr.log
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_a2.py --continue-from a2 --manifest
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--dry-run', metavar='SCRATCH_DIR', default=None)
    g.add_argument('--manifest', action='store_true')
    ap.add_argument('--continue-from', choices=('a2',), default=None,
                    help="ruling (a): run the five solves a2 did not run, into P516TPERF/a2b/ (TSO c1 by reference)")
    return ap.parse_args(argv)


ARGS = _parse_args()

# Import the Part A harness for its solve path / timing / option assembly / log handling. Its module-level argument
# parse is given `--manifest`, so it installs NO guard of its own; this harness installs the guard below, before any
# production import, and points the imported module's `_GUARD` at it (run_case reads that global).
import hashlib as _hashlib  # noqa: E402


def _file_sha(path):
    with open(path, 'rb') as handle:
        return _hashlib.sha256(handle.read()).hexdigest()


# The Part A harness is imported from the WORKING TREE (it may be under edit by another Worker): its bytes are hashed
# immediately before and after the import; a change across the import refuses (the imported bytes would be unknown).
_TP_SHA_BEFORE_IMPORT = _file_sha(os.path.join(REPO, 'p516_tperf_profile.py'))
_argv = sys.argv
sys.argv = [os.path.join(REPO, 'p516_tperf_profile.py'), '--manifest']
import p516_tperf_profile as tp  # noqa: E402  (sets the production thread caps, chdir REPO)
sys.argv = _argv
_TP_SHA_AFTER_IMPORT = _file_sha(tp.THIS)
if _TP_SHA_AFTER_IMPORT != _TP_SHA_BEFORE_IMPORT:
    raise RuntimeError(f'p516_tperf_profile.py changed across the import ({_TP_SHA_BEFORE_IMPORT} -> '
                       f'{_TP_SHA_AFTER_IMPORT}); imported bytes unknown -- refusing')
if tp._GUARD is not None:
    raise RuntimeError('imported Part A harness installed a guard (expected none)')
if 'network' in sys.modules:
    raise RuntimeError('a production module was imported before the guard')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PERMITTED = (('network.py', '_run_smopf_solver_attempt'),)
CONTINUE = ARGS.continue_from == 'a2'
DECLARED_SOLVES = 5 if CONTINUE else 6
TAG = 'A2b' if CONTINUE else 'A2'
if ARGS.manifest:
    _GUARD = None
elif ARGS.dry_run:
    _GUARD = SolveProfileGuard((), label=f'P5.16 T-perf {TAG} DRY RUN (zero solves)').install()
else:
    _GUARD = SolveProfileGuard(PERMITTED, label=f'P5.16 T-perf {TAG} ({DECLARED_SOLVES} declared solves)').install()
tp._GUARD = _GUARD

import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402

import psutil  # noqa: E402

THIS = os.path.abspath(__file__)
STAGE = 'P5.16 T-perf A2 -- warm-start barrier A/B (mu_init) on the certified SRP1 unit-cell blocks'
AUTHORITY = ["author 'Decision 4' (2026-10-31): diagnostic A/B, <= 6 solves, before expert R7",
             'P5_16_TPERF_SPEC_v1.md', 'TASKS.md T-perf / Expert R7 lines',
             "Planner ruling (b), 2026-10-31 (after the stopped attempt's comparator finding)"]
A2_ROOT_REL = os.path.join(tp.TPERF_ROOT_REL, 'a2')
OUT_REL = os.path.join(tp.TPERF_ROOT_REL, 'a2b') if CONTINUE else A2_ROOT_REL
_LOG_STEM = 'a2b' if CONTINUE else 'a2'
LAUNCH_LOGS_REL = [os.path.join(tp.TPERF_ROOT_REL, f'{_LOG_STEM}_launch_stdout.log'),
                   os.path.join(tp.TPERF_ROOT_REL, f'{_LOG_STEM}_launch_stderr.log')]
FORBIDDEN_OUTPUT_ROOTS = (os.path.join('data', 'SRP1', 'Results', 'P516D0'), tp.LOCAL_OUT_REL, tp.STUDIO_OUT_REL,
                          os.path.join(tp.TPERF_ROOT_REL, 'azero')) + (
    (A2_ROOT_REL, os.path.join(tp.TPERF_ROOT_REL, 'studio_dry'), os.path.join(tp.TPERF_ROOT_REL, 'studio_dry_equiv'))
    if CONTINUE else ())

MU_INIT = 1e-6
PERTURB_FACTOR = 10.0
OBJ_REL_GATE = 1e-9
VAR_ABS_GATE = 1e-6
BLOCKS = ('tso', 'dso')
CASES = (('c1_ident_mu', False, True), ('c2_pert', True, False), ('c3_pert_mu', True, True))  # (label, perturbed, mu)
CASE_PERTURBED = {c: p for c, p, _ in CASES}
CASE_MU = {c: m for c, _, m in CASES}
TARGET_PARAMS = ('vmag_req', 'p_pf_req', 'q_pf_req', 'p_ess_req', 'q_ess_req')
PARAM_CHANNEL = {'vmag_req': 'v', 'p_pf_req': 'pf_p', 'q_pf_req': 'pf_q', 'p_ess_req': 'ess_p', 'q_ess_req': 'ess_q'}
BLOCK_NODES = {'tso': (5, 7, 9), 'dso': (tp.DSO_NODE,)}
ESS_CHANNEL = {('tso', 'ess_p'): 'ess_p_tso_esso', ('tso', 'ess_q'): 'ess_q_tso_esso',
               ('dso', 'ess_p'): 'ess_p_dso_esso', ('dso', 'ess_q'): 'ess_q_dso_esso'}
NETWORK_NAME = {'tso': 'case9', 'dso': 'case33_2'}

T01B_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01b')
COUPLING_REL = os.path.join(T01B_ROOT_REL, 'coupling.json')
T01B_MANIFEST_REL = os.path.join(T01B_ROOT_REL, 'manifest_sha256.json')
T01B_MANIFEST_SHA_PREFIX = '4e5bbd45'   # committed in 0b92e24a (TASKS.md: "evidence manifest 4e5bbd45")
T01B_CELL = 'unit_n7'
PART_A_ROOT_REL = tp.LOCAL_OUT_REL
PART_A_MANIFEST_REL = os.path.join(PART_A_ROOT_REL, 'manifest_sha256.json')
PART_A_MANIFEST_SHA_PREFIX = 'fbfd6310'  # P5_16_TPERF_REPORT.md
PART_A_JSON_REL = os.path.join(PART_A_ROOT_REL, 'tperf_local.json')
PART_A_WARM_LOG_REL = {b: os.path.join(PART_A_ROOT_REL, 'logs', f'tperf_{b}_i_warm_2025_Spring.log') for b in BLOCKS}
# Planner ruling (b), 2026-10-31: the c1 objective comparator = Part A's warm limit, as its log prints it (17 digits)
RULING = "Planner ruling (b), 2026-10-31: pickled (pre-tail) options for all six solves; c1 objective gated on Part A's warm limit; c1 Vars reported vs certified, not gated; c3 vs c2 gated on objective and every Var"
PART_A_WARM_LIMIT_DECLARED = {'tso': -52244.700129902631, 'dso': 54561.135089701755}
OPTION_KEYS_EXCLUDED_FROM_REGIME_MATCH = ('output_file', 'mu_init')

# ---- continuation A2b (Planner ruling (a), 2026-10-31) ----
RULING_A = ("Planner ruling (a), 2026-10-31: the six declared solves span two launches, a2 (1 solve, done) + a2b (the "
            "five unrun; verify(5) exact); objective gates restated to the stopping-slack basis (|df| <= max over the "
            "two solves of the terminal unscaled complementarity; ratio < 1 = indeterminate (inside stopping slack) = "
            "pass, >= 1 = real difference); 1e-9 relative reported beside, not gated; c3 vs c2 Vars: agreement iff max "
            "|dx| <= 1e-4 p.u., reported not stopped; c1-gate stop removed (stops: failed solve, restoration, hash "
            "mismatch); TSO c1 not re-run")
SPLIT = {'declared_total': 6, 'a2': {'solves': ['tso_c1_ident_mu'], 'count': 1, 'status': 'done (stopped run)',
                                     'manifest_sha256': 'c22e94d3ac566783afcbeafa68a4d3a25e1f4a517149f027c390a58183be8d72',
                                     'a2_json_sha256': 'e87d67b2aefe053ba1deec04f237f0c673530fdec6adc120b7585e6d8bdb178b'},
         'a2b': {'solves': ['tso_c2_pert', 'tso_c3_pert_mu', 'dso_c1_ident_mu', 'dso_c2_pert', 'dso_c3_pert_mu'],
                 'count': 5, 'guard': 'bounded SolveProfileGuard, verify(5) exact'}}
A2_MANIFEST_REL = os.path.join(A2_ROOT_REL, 'manifest_sha256.json')
A2_JSON_REL = os.path.join(A2_ROOT_REL, 'a2.json')
A2_C1_LOG_REL = os.path.join(A2_ROOT_REL, 'logs', 'a2_tso_c1_ident_mu_2025_Spring.log')
A2_HARNESS_SHA = '019d89a9db85d68627211835df4326ec17e95d5391499c97d020662df981df51'   # a2 manifest, p516_tperf_a2.py
A2_HARNESS_GIT_BLOB = '47834395a966e125d8bcbcf8e84205658727073e'   # the staged blob of those bytes (git ls-files -s)
TP_COMMITTED_SHA = 'd013452fce3258b03771e1dba178ec8802fe119622553dc6d08dd472b39dfd9a'   # P5_16_TPERF_REPORT.md
SKIPPED_CASES = ('tso_c1_ident_mu',) if CONTINUE else ()
VAR_AGREE_ABS = 1e-4
TP_FUNCTIONS_USED = ('run_case', 'Instrument', 'parse_ipopt_log', '_post_solve_parse', 'Lock', 'verify_inputs',
                     '_write_json', 'capture_checklist', 'expected_log_path', '_build_time_objects',
                     '_structure_signature', '_norm', '_sha', '_git', '_git_free_cmd', '_utc')

PREDICTIONS = {
    'recorded': 'before any solve (task text, Planner, 2026-10-31); also written to predictions.json before solve 1',
    'A1': {'text': 'identical problem with mu_init 1e-6: TSO <= 8 and DSO <= 15 iterations (from 22 / 51)',
           'tso_max_iterations': 8, 'dso_max_iterations': 15},
    'A2': {'text': "(2) within +-30 % of Part A's warm counts", 'part_a_warm': {'tso': 22, 'dso': 51},
           'rel_band': 0.30},
    'A3': {'text': '(3) at least 40 % fewer iterations than (2)', 'min_reduction': 0.40},
    'A4': {'text': 'all solution agreements within the gate (same optimum, no restoration)',
           'obj_rel_gate': OBJ_REL_GATE, 'var_abs_gate': VAR_ABS_GATE,
           'gate_scope_per_ruling_b': ("c1: objective vs Part A's warm limit only (Vars reported vs certified, not "
                                       'gated); c3 vs c2: objective and every free Var')},
    'A5': {'text': 'wall-time reduction of (3) vs (2) >= 30 % on the DSO (run_smopf wall)', 'min_reduction': 0.30},
}

T0 = time.time()
_RUN_LOG = None


def _log(msg):
    line = f'{time.strftime("%H:%M:%S")} [{TAG} +{time.time() - T0:8.2f}s] {msg}'
    print(line, flush=True)
    if _RUN_LOG is not None:
        with open(_RUN_LOG, 'a') as handle:
            handle.write(line + '\n')


# ======================================================================================================================
#  inputs beyond Part A's: T0.1b coupling evidence, Part A baseline, the certifying options
# ======================================================================================================================
def verify_side_inputs():
    out = {}
    man_sha = tp._sha(os.path.join(REPO, T01B_MANIFEST_REL))
    if not man_sha.startswith(T01B_MANIFEST_SHA_PREFIX):
        raise RuntimeError(f'T0.1b manifest hash {man_sha} does not start with {T01B_MANIFEST_SHA_PREFIX}')
    man = json.load(open(os.path.join(REPO, T01B_MANIFEST_REL)))
    pinned = man['files']['coupling.json']['sha256']
    now = tp._sha(os.path.join(REPO, COUPLING_REL))
    if now != pinned:
        raise RuntimeError(f'coupling.json hash {now} != T0.1b manifest {pinned}')
    out['coupling'] = {'path': COUPLING_REL, 'sha256': now, 'manifest': T01B_MANIFEST_REL, 'manifest_sha256': man_sha}
    pa_man_sha = tp._sha(os.path.join(REPO, PART_A_MANIFEST_REL))
    if not pa_man_sha.startswith(PART_A_MANIFEST_SHA_PREFIX):
        raise RuntimeError(f'Part A manifest hash {pa_man_sha} does not start with {PART_A_MANIFEST_SHA_PREFIX}')
    pa_files = json.load(open(os.path.join(REPO, PART_A_MANIFEST_REL)))['files']
    out['part_a'] = {'manifest': PART_A_MANIFEST_REL, 'manifest_sha256': pa_man_sha, 'files': {}}
    for rel in [PART_A_JSON_REL] + list(PART_A_WARM_LOG_REL.values()):
        now = tp._sha(os.path.join(REPO, rel))
        if pa_files.get(rel) != now:
            raise RuntimeError(f'Part A file {rel} hash {now} != its manifest {pa_files.get(rel)}')
        out['part_a']['files'][rel] = now
    return out


def channel_magnitudes(coupling):
    """Per block, per channel: max |terminal inter-copy mismatch| over the block's 2025 Spring entries (physical units)."""
    rows = [r for r in coupling['mismatch_step1'][T01B_CELL] if r['year'] == tp.YEAR and r['day'] == tp.DAY]
    consts = {k: v for k, v in coupling['constants'][T01B_CELL].items() if k.startswith(f'{tp.YEAR}_{tp.DAY}_')}
    out = {}
    for block in BLOCKS:
        nodes = BLOCK_NODES[block]
        sel = [r for r in rows if r['node'] in nodes]
        mags = {}

        def amax(chan, f=lambda r: abs(r['mismatch'])):
            rs = [r for r in sel if r['channel'] == chan]
            if len(rs) != 24 * len(nodes):
                raise RuntimeError(f'{block} {chan}: {len(rs)} rows (expected {24 * len(nodes)})')
            best = max(rs, key=f)
            return f(best), {'node': best['node'], 'period': best['period'], 'n_rows': len(rs), 'unit': best['unit']}

        # v: from v_W rows (a = TSO side, b = DSO side, kV^2): magnitude dV = sqrt(a) - sqrt(b) [kV]
        for r in sel:
            if r['channel'] == 'v_W' and abs((r['a_phys'] - r['b_phys']) - r['mismatch']) > 1e-9 * max(1.0, abs(r['a_phys'])):
                raise RuntimeError(f'v_W row inconsistent: {r}')
        dv, arg = amax('v_W', lambda r: abs(math.sqrt(r['a_phys']) - math.sqrt(r['b_phys'])))
        arg['unit'] = 'kV (magnitude, |sqrt(a_phys) - sqrt(b_phys)| of the v_W row)'
        mags['v'] = {'max_abs': dv, **arg}
        for ch in ('pf_p', 'pf_q'):
            v, arg = amax(ch)
            mags[ch] = {'max_abs': v, **arg}
        for ch in ('ess_p', 'ess_q'):
            src = ESS_CHANNEL[(block, ch)]
            v, arg = amax(src)
            mags[ch] = {'max_abs': v, 'source_channel': src, **arg}
        c = [consts[f'{tp.YEAR}_{tp.DAY}_{n}'] for n in nodes]
        if block == 'tso':
            vb = {x['vb_tso_kV'] for x in c}
            sb = {x['sb_tso_MVA'] for x in c}
        else:
            vb = {x['vb_dso_kV'] for x in c}
            sb = {x['sb_dso_MVA'] for x in c}
        if len(vb) != 1 or len(sb) != 1:
            raise RuntimeError(f'{block}: non-unique bases {vb} {sb}')
        out[block] = {'nodes': list(nodes), 'channels': mags, 'v_base_kV': vb.pop(), 's_base_MVA': sb.pop(),
                      'constants': {str(n): x for n, x in zip(nodes, c)}}
    return out


def perturbation_deltas(mags_block):
    vb, sb = mags_block['v_base_kV'], mags_block['s_base_MVA']
    out = {}
    for pname in TARGET_PARAMS:
        ch = PARAM_CHANNEL[pname]
        phys = PERTURB_FACTOR * mags_block['channels'][ch]['max_abs']
        base = vb if ch == 'v' else sb
        out[pname] = {'channel': ch, 'shift_physical': phys,
                      'unit_physical': 'kV' if ch == 'v' else ('Mvar' if ch.endswith('_q') else 'MW'),
                      'base': base, 'shift_param_units': phys / base,
                      'param_units': 'pu of v_base (vmag)' if ch == 'v' else 'pu of s_base'}
    return out


def apply_perturbation(model, deltas, network_module, pe):
    before = network_module.capture_block_mutable_state(model)
    applied = {}
    for pname, d in deltas.items():
        comp = getattr(model, pname)
        if not comp.mutable:
            raise RuntimeError(f'{pname} is not mutable')
        n = 0
        for idx in comp:
            comp[idx].set_value(comp[idx].value + d['shift_param_units'])
            n += 1
        applied[pname] = n
    after = network_module.capture_block_mutable_state(model)
    # every other mutable Param, every Var value / raw bound / fixed flag, the active flags and the suffixes unchanged
    # (bounds and Params compared in tp._structure_signature's normalised form: a raw Pyomo expression would turn
    # `==` into a relation)
    sig_b, sig_a = tp._structure_signature(before), tp._structure_signature(after)
    changed_params = sorted(n for n in sig_b['params'] if sig_b['params'][n] != sig_a['params'][n])
    if changed_params != sorted(deltas):
        raise RuntimeError(f'perturbation changed {changed_params}, expected exactly {sorted(deltas)}')
    max_dev = 0.0
    for pname, d in deltas.items():
        for idx, v0 in before['params'][pname].items():
            max_dev = max(max_dev, abs((after['params'][pname][idx] - v0) - d['shift_param_units']))
    values_b = {n: {i: tp._norm(e[0]) for i, e in dd.items()} for n, dd in before['vars'].items()}
    values_a = {n: {i: tp._norm(e[0]) for i, e in dd.items()} for n, dd in after['vars'].items()}
    others_same = bool(values_b == values_a and sig_b['vars'] == sig_a['vars']
                       and sig_b['constraints'] == sig_a['constraints'] and sig_b['objectives'] == sig_a['objectives'])
    if not others_same:
        raise RuntimeError('perturbation changed Var values / bounds / fixed flags or active flags')
    if max_dev > 1e-15:
        raise RuntimeError(f'applied shift deviates from the declared by {max_dev}')
    return {'entries_shifted': applied, 'changed_params': changed_params,
            'max_abs_deviation_from_declared_shift': max_dev, 'vars_constraints_objectives_unchanged': others_same,
            'suffixes_unchanged': _suffix_equal(before['suffixes'], after['suffixes'])}


def _suffix_equal(a, b):
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return None


def certification_options():
    """ZERO-SOLVE: the IPOPT options in force at the certifying cycle's primary solves of the two blocks (production
    per-attempt records of this cell; hash pinned by both manifests via tp.verify_inputs)."""
    path = os.path.join(REPO, tp.INPUTS['production_solve_records'][0])
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    out = {}
    k = tp.INSTANCE['certification_cycle']
    for block in BLOCKS:
        sel = [r for r in rows if r['network'] == NETWORK_NAME[block] and r['year'] == tp.YEAR and r['day'] == tp.DAY
               and r['attempt'] == 'primary']
        cert = [r for r in sel if r['round'] == k]
        if len(cert) != 1:
            raise RuntimeError(f'{block}: {len(cert)} primary records at round {k}')
        changes, prev = [], object()
        for r in sel:
            key = (r['compl_inf_tol_passed'], r['tol_in_force'], r['mu_floor'])
            if key != prev:
                changes.append({'round': r['round'], 'compl_inf_tol_passed': r['compl_inf_tol_passed'],
                                'tol_in_force': r['tol_in_force'], 'mu_floor': r['mu_floor']})
                prev = key
        c = cert[0]
        out[block] = {'certifying_round': k, 'compl_inf_tol_passed': c['compl_inf_tol_passed'],
                      'compl_inf_tol_in_force': c['compl_inf_tol_in_force'], 'tol_in_force': c['tol_in_force'],
                      'mu_final': c['mu_final'], 'mu_floor': c['mu_floor'], 'iterations': c['iterations'],
                      'warm_start': c['warm_start'], 'exit': c['exit'], 'option_regime_changes': changes}
    out['source'] = tp.INPUTS['production_solve_records'][0]
    return out


_FINAL_OBJ_RE = re.compile(r'Objective\.+:\s+(\S+)\s+(\S+)')


def log_final_objective(path):
    if not os.path.exists(path):
        return None
    objs = _FINAL_OBJ_RE.findall(open(path).read())
    return float(objs[-1][1]) if objs else None


def part_a_warm_limit(block):
    text = open(os.path.join(REPO, PART_A_WARM_LOG_REL[block])).read()
    objs = _FINAL_OBJ_RE.findall(text)
    return {'final_objective_unscaled': float(objs[-1][1]), 'log': PART_A_WARM_LOG_REL[block]}


def _regime(options, excluded):
    return {k: v for k, v in (options or {}).items() if k not in excluded}


def comparator_check(cert_obj, cert_opts, passed_opts, part_a_record):
    """Ruling (b) pre-launch check (zero solves). Gated: (i) the c1 objective comparator parsed from Part A's warm log
    equals the ruling's declared 17-digit value exactly; (ii) c1's options (less output_file, mu_init) equal Part A's
    warm (i) options (less output_file). Recorded only: the certifying options and the certified-vs-limit offset."""
    out = {'ruling': RULING, 'obj_rel_gate': OBJ_REL_GATE, 'blocks': {}}
    failures = []
    for block in BLOCKS:
        pa = part_a_warm_limit(block)
        f_lim = pa['final_objective_unscaled']
        declared = PART_A_WARM_LIMIT_DECLARED[block]
        prep = part_a_record['solves'][f'{block}_i_warm']['preparation']
        pa_opts = dict(prep['production_options'])
        pa_opts.update(prep['options_added_or_changed'])
        pa_regime = _regime(pa_opts, ('output_file',))
        a2_regime = _regime(passed_opts[block], OPTION_KEYS_EXCLUDED_FROM_REGIME_MATCH)
        f_cert = cert_obj[block]
        rel = abs(f_lim - f_cert) / abs(f_cert)
        row = {'c1_objective_comparator (Part A warm limit, log final unscaled)': f_lim,
               'declared_by_ruling': declared,
               'parsed_equals_declared_exactly': bool(f_lim == declared),
               'part_a_log': pa['log'],
               'part_a_warm_options_less_output_file': pa_regime,
               'c1_options_less_output_file_and_mu_init': a2_regime,
               'option_regime_matches_part_a_warm': bool(pa_regime == a2_regime),
               'option_regime_differences': {k: [pa_regime.get(k), a2_regime.get(k)]
                                             for k in sorted(set(pa_regime) | set(a2_regime))
                                             if pa_regime.get(k) != a2_regime.get(k)},
               'recorded_not_gated': {
                   'certified_objective (pe.value at the pickled point)': f_cert,
                   'part_a_limit_minus_certified': f_lim - f_cert, 'part_a_limit_rel_diff': rel,
                   'part_a_limit_rel_diff_over_gate': rel / OBJ_REL_GATE,
                   'compl_inf_tol_passed_this_run': passed_opts[block].get('compl_inf_tol'),
                   'compl_inf_tol_passed_at_certification': cert_opts[block]['compl_inf_tol_passed'],
                   'mu_floor_at_certification': cert_opts[block]['mu_floor']}}
        if not row['parsed_equals_declared_exactly']:
            failures.append(f'{block}: Part A limit parsed {f_lim!r} != declared {declared!r}')
        if not row['option_regime_matches_part_a_warm']:
            failures.append(f'{block}: option regime differs from Part A warm: {row["option_regime_differences"]}')
        out['blocks'][block] = row
    out['failures'] = failures
    out['verdict'] = 'ok' if not failures else 'REFUSE: ' + '; '.join(failures)
    return out


def compare_c1(sol, block, certified):
    """Ruling (b): c1's objective gated on Part A's warm limit; its Vars reported vs the certified values, not gated."""
    f_lim = PART_A_WARM_LIMIT_DECLARED[block]
    fa = sol['objective']
    rel = abs(fa - f_lim) / abs(f_lim)
    vs_cert = compare(sol, certified, f'{block}_c1_ident_mu vs certified values (REPORTED, NOT GATED)')
    for k in ('objective_within_gate', 'vars_within_gate', 'gate_pass'):
        vs_cert[f'{k} (informational only; not a gate under ruling (b))'] = vs_cert.pop(k)
    return {'label': f'{block}_c1_ident_mu objective vs Part A warm limit (ruling (b))',
            'objective_a': fa, 'objective_b (Part A warm limit, log 17 digits)': f_lim,
            'objective_diff': fa - f_lim, 'objective_rel_diff': rel,
            'objective_within_gate': bool(rel <= OBJ_REL_GATE), 'gate_pass': bool(rel <= OBJ_REL_GATE),
            'gate': {'objective_rel': OBJ_REL_GATE, 'vars': 'not gated (ruling (b))'},
            'vars_vs_certified_reported_not_gated': vs_cert}


# ======================================================================================================================
#  continuation A2b (ruling (a)): a2 by reference, terminal statistics, the stopping-slack gate, import provenance
# ======================================================================================================================
_FINAL_STAT_RE = {
    'objective': re.compile(r'^Objective\.+:\s+(\S+)\s+(\S+)\s*$', re.M),
    'dual_infeasibility': re.compile(r'^Dual infeasibility\.+:\s+(\S+)\s+(\S+)\s*$', re.M),
    'constraint_violation': re.compile(r'^Constraint violation\.+:\s+(\S+)\s+(\S+)\s*$', re.M),
    'variable_bound_violation': re.compile(r'^Variable bound violation:\s+(\S+)\s+(\S+)\s*$', re.M),
    'complementarity': re.compile(r'^Complementarity\.+:\s+(\S+)\s+(\S+)\s*$', re.M),
    'overall_nlp_error': re.compile(r'^Overall NLP error\.+:\s+(\S+)\s+(\S+)\s*$', re.M),
}


def final_statistics(path):
    """IPOPT's final statistics block: {name: {'scaled', 'unscaled'}}, read from the text AFTER the (unique) "Number of
    Iterations" line -- at file_print_level >= 6 every iteration prints the same-named lines, so only the block that
    follows that line is the terminal one. Exactly one such line (one solve per log) and one of each stat after it."""
    text = open(path).read()
    heads = list(re.finditer(r'^Number of Iterations\.+:', text, re.M))
    if len(heads) != 1:
        raise RuntimeError(f'{path}: {len(heads)} "Number of Iterations" lines (expected 1)')
    tail = text[heads[0].start():]
    out = {'log': os.path.relpath(path, REPO), 'n_final_blocks': len(heads)}
    for k, rx in _FINAL_STAT_RE.items():
        m = rx.findall(tail)
        if len(m) != 1:
            raise RuntimeError(f'{path}: {len(m)} "{k}" lines after "Number of Iterations" (expected 1)')
        out[k] = {'scaled': float(m[0][0]), 'unscaled': float(m[0][1])}
    return out


def slack_gate(label, f_a, f_b, stats_a, stats_b, name_a, name_b):
    """Ruling (a): |df| <= max over the two solves of the terminal UNSCALED complementarity. ratio < 1 ->
    'indeterminate (inside stopping slack)' = pass; >= 1 -> 'real difference'. 1e-9 relative reported beside."""
    df = f_a - f_b
    ca, cb = stats_a['complementarity']['unscaled'], stats_b['complementarity']['unscaled']
    bar = max(ca, cb)
    ratio = abs(df) / bar
    rel = abs(df) / abs(f_b)
    return {'label': label, 'f_a': f_a, 'f_b': f_b, 'a': name_a, 'b': name_b, 'delta_f': df, 'abs_delta_f': abs(df),
            'terminal_unscaled_complementarity': {name_a: ca, name_b: cb},
            'bar (max terminal unscaled complementarity)': bar, 'ratio_abs_delta_f_over_bar': ratio,
            'verdict': ('indeterminate (inside stopping slack)' if ratio < 1 else 'real difference'),
            'pass': bool(ratio < 1),
            'recorded_not_gated': {
                'terminal_unscaled_dual_infeasibility': {name_a: stats_a['dual_infeasibility']['unscaled'],
                                                         name_b: stats_b['dual_infeasibility']['unscaled']},
                'terminal_constraint_violation': {name_a: stats_a['constraint_violation']['unscaled'],
                                                  name_b: stats_b['constraint_violation']['unscaled']},
                'objective_rel_diff': rel, 'objective_rel_diff_over_1e-9 (reported beside, not gated)': rel / OBJ_REL_GATE},
            'gate': 'ruling (a): |df| <= max(terminal unscaled complementarity of the two solves); ratio < 1 passes'}


def verify_a2_reference(out_dir):
    """ZERO-SOLVE. a2's manifest, a2.json and c1's IPOPT log hash-verified (manifest against the ruling's value, files
    against the manifest); a2's harness bytes recovered from the staged git blob and checked against the manifest,
    then written into the a2b root for preservation. Returns the a2 TSO c1 record by reference."""
    man_sha = tp._sha(os.path.join(REPO, A2_MANIFEST_REL))
    if man_sha != SPLIT['a2']['manifest_sha256']:
        raise RuntimeError(f'a2 manifest hash {man_sha} != {SPLIT["a2"]["manifest_sha256"]}')
    files = json.load(open(os.path.join(REPO, A2_MANIFEST_REL)))['files']
    checked = {}
    for rel in (A2_JSON_REL, A2_C1_LOG_REL):
        now = tp._sha(os.path.join(REPO, rel))
        if files.get(rel) != now:
            raise RuntimeError(f'a2 file {rel} hash {now} != a2 manifest {files.get(rel)}')
        checked[rel] = now
    if checked[A2_JSON_REL] != SPLIT['a2']['a2_json_sha256']:
        raise RuntimeError(f'a2.json hash {checked[A2_JSON_REL]} != ruling {SPLIT["a2"]["a2_json_sha256"]}')
    if files.get('p516_tperf_a2.py') != A2_HARNESS_SHA:
        raise RuntimeError(f'a2 manifest harness hash {files.get("p516_tperf_a2.py")} != {A2_HARNESS_SHA}')
    blob = subprocess.run(['git', 'cat-file', '-p', A2_HARNESS_GIT_BLOB], cwd=REPO, capture_output=True).stdout
    blob_sha = _hashlib.sha256(blob).hexdigest()
    if blob_sha != A2_HARNESS_SHA:
        raise RuntimeError(f'git blob {A2_HARNESS_GIT_BLOB} sha256 {blob_sha} != a2 harness {A2_HARNESS_SHA}')
    copy_path = os.path.join(out_dir, f'a2_harness_as_run_{A2_HARNESS_SHA[:8]}.py.txt')
    with open(copy_path, 'xb') as handle:
        handle.write(blob)
    a2 = json.load(open(os.path.join(REPO, A2_JSON_REL)))
    c1 = a2['solves']['tso_c1_ident_mu']
    return {'split': SPLIT, 'a2_manifest': A2_MANIFEST_REL, 'a2_manifest_sha256': man_sha, 'files_verified': checked,
            'a2_harness_sha256': A2_HARNESS_SHA, 'a2_harness_git_blob': A2_HARNESS_GIT_BLOB,
            'a2_harness_copy': {'path': os.path.relpath(copy_path, REPO), 'sha256': tp._sha(copy_path)},
            'a2_status': a2['status'], 'a2_stopped': a2['stopped'], 'a2_guard': a2['guard'],
            'a2_provenance': a2['provenance'],
            'a2_part_a_harness_sha256_imported': a2['provenance']['part_a_harness_sha256'],
            'tso_c1_summary_row': a2['summary']['tso_c1_ident_mu'],
            'tso_c1_mu_trace': c1['mu_trace'], 'tso_c1_solver_options_passed': c1['solver_options_passed'],
            'tso_c1_timing': c1['timing'], 'tso_c1_preparation': c1['preparation'],
            'tso_c1_agreement_in_a2 (ruling (b) form)': a2['agreement']['tso_c1_vs_part_a_limit'],
            'tso_c1_final_statistics': final_statistics(os.path.join(REPO, A2_C1_LOG_REL))}


def tp_import_provenance():
    """ZERO-SOLVE. Which p516_tperf_profile.py bytes this run imported, against the committed (HEAD) version and the
    version a2 imported; per used function / class: source identical to HEAD's?"""
    import ast
    head_src = subprocess.run(['git', 'show', 'HEAD:p516_tperf_profile.py'], cwd=REPO, capture_output=True,
                                 text=True).stdout
    work_src = open(tp.THIS).read()

    def defs(src):
        tree = ast.parse(src)
        return {n.name: ast.get_source_segment(src, n) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in TP_FUNCTIONS_USED}
    h, w = defs(head_src), defs(work_src)
    return {'imported_sha256_before_import': _TP_SHA_BEFORE_IMPORT, 'imported_sha256_after_import': _TP_SHA_AFTER_IMPORT,
            'imported_sha256_at_record_time': tp._sha(tp.THIS),
            'head_sha256': _hashlib.sha256(head_src.encode()).hexdigest(), 'committed_sha256_part_a_report': TP_COMMITTED_SHA,
            'a2_imported_sha256': 'b8dbb6aa36c87e538039922cd9932d75aad336ef3a5918363dc7fc79890ca87d',
            'git_status': tp._git(['status', '--porcelain', '--', 'p516_tperf_profile.py']),
            'used_definitions_identical_to_head': {n: (n in h and n in w and h[n] == w[n]) for n in TP_FUNCTIONS_USED},
            'note': ('recorded provenance mismatch, not a fault (ruling (a)): the import is the working tree under edit '
                     'by the Part B Worker')}


# ======================================================================================================================
#  measurement helpers
# ======================================================================================================================
_ITER_LINE_RE = re.compile(r'^\s*(\d+)(r?)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+')
_MU_LINE_RE = re.compile(r'Current barrier parameter mu = (\S+)')


def mu_trace(text):
    seen = {}
    for line in text.splitlines():
        m = _ITER_LINE_RE.match(line)
        if m and m.group(6) not in ('lg(mu)',):
            key = (int(m.group(1)), m.group(2))
            if key not in seen:
                try:
                    seen[key] = float(m.group(6))
                except ValueError:
                    continue
    it0 = seen.get((0, ''))
    mus = [float(x) for x in _MU_LINE_RE.findall(text)]
    seq = []
    for (_, r), lg in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if not seq or seq[-1] != lg:
            seq.append(lg)
    return {'lg_mu_iter0_table': it0, 'mu_iter0_table': (10.0 ** it0) if it0 is not None else None,
            'mu_first_current_barrier_line': mus[0] if mus else None,
            'mu_last_current_barrier_line': mus[-1] if mus else None,
            'lg_mu_distinct_sequence': seq, 'n_iteration_lines_dedup': len(seen),
            'lg_mu_per_iteration': [[k, r, lg] for (k, r), lg in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1]))]}


def primal_snapshot(model, pe):
    vals = {}
    for v in model.component_data_objects(pe.Var, descend_into=True):
        if v.fixed:
            continue
        vals[v.name] = v.value
    objs = [o for o in model.component_data_objects(pe.Objective, active=True)]
    if len(objs) != 1:
        raise RuntimeError(f'{len(objs)} active objectives')
    return {'objective_name': objs[0].name, 'objective': pe.value(objs[0]), 'vars': vals}


def compare(a, b, label):
    """a vs comparator b."""
    fa, fb = a['objective'], b['objective']
    obj_rel = abs(fa - fb) / abs(fb) if fb else abs(fa - fb)
    keys = set(a['vars']) | set(b['vars'])
    missing = sorted(k for k in keys if k not in a['vars'] or k not in b['vars'])
    none = sorted(k for k in keys if k not in missing and (a['vars'][k] is None or b['vars'][k] is None))
    diffs = []
    for k in keys:
        if k in missing or k in none:
            continue
        diffs.append((abs(a['vars'][k] - b['vars'][k]), k))
    diffs.sort(reverse=True)
    per_comp = {}
    for d, k in diffs:
        c = k.split('[', 1)[0]
        if d > per_comp.get(c, (-1.0, None))[0]:
            per_comp[c] = (d, k)
    over = [(d, k) for d, k in diffs if d > VAR_ABS_GATE]
    over_comp = {}
    for d, k in over:
        c = k.split('[', 1)[0]
        over_comp[c] = over_comp.get(c, 0) + 1
    max_abs = diffs[0][0] if diffs else None
    obj_ok = obj_rel <= OBJ_REL_GATE
    var_ok = (not over) and (not missing) and (not none)
    return {'label': label, 'objective_a': fa, 'objective_b': fb, 'objective_diff': fa - fb, 'objective_rel_diff': obj_rel,
            'objective_within_gate': bool(obj_ok), 'n_vars_compared': len(diffs), 'n_missing': len(missing),
            'missing_sample': missing[:10], 'n_value_none': len(none), 'none_sample': none[:10],
            'max_abs_var_diff': max_abs, 'n_vars_over_gate': len(over),
            'over_gate_by_component': dict(sorted(over_comp.items(), key=lambda kv: -kv[1])),
            'worst_20': [{'var': k, 'abs_diff': d, 'a': a['vars'][k], 'b': b['vars'][k]} for d, k in diffs[:20]],
            'max_by_component': {c: {'abs_diff': d, 'var': k} for c, (d, k) in sorted(per_comp.items(), key=lambda kv: -kv[1][0])},
            'vars_within_gate': bool(var_ok), 'gate_pass': bool(obj_ok and var_ok),
            'gate': {'objective_rel': OBJ_REL_GATE, 'var_abs': VAR_ABS_GATE, 'vars': 'every free (unfixed) Pyomo Var'}}


def solve_row(rec):
    s = (rec.get('ipopt_log') or {}).get('summary', {})
    t = rec['timing']
    return {'termination': rec['termination_condition'], 'succeeded': rec['succeeded'],
            'iterations': s.get('iterations'), 'restoration_iterations': s.get('restoration_iteration_lines'),
            'restoration_phases': s.get('restoration_phases'), 'exit': s.get('exit'),
            'mu_iter0_table': (rec.get('mu_trace') or {}).get('mu_iter0_table'),
            'mu_first_current_barrier_line': (rec.get('mu_trace') or {}).get('mu_first_current_barrier_line'),
            'mu_final': (rec.get('production_attempt_parse') or {}).get('mu_final'),
            'mu_floor': (rec.get('production_attempt_parse') or {}).get('mu_floor'),
            'floor_status': (rec.get('production_attempt_parse') or {}).get('floor_status'),
            'run_smopf_s': t['run_smopf_wall_s'], 'nl_write_s': t['nl_write_wall_s (NLWriter.__call__)'],
            'result_load_s': t['result_load_wall_s (sol read + load_from)'],
            'ipopt_process_s': t['ipopt_process_wall_s (_execute_command)'],
            'ipopt_overall_algorithm_s': s.get('overall_algorithm_wall_s'),
            'ipopt_linear_solver_s': s.get('linear_solver_wall_s'),
            'ipopt_function_evals_s': s.get('function_evaluations_wall_s'),
            'mu_init_passed': (rec.get('solver_options_passed') or {}).get('mu_init')}


def evaluate_predictions(record, rows=None, gates=None):
    """a2 mode: rows = record['summary'], gates = each agreement's gate_pass (unchanged). A2b: rows = all six solves
    (TSO c1 from a2 by reference), gates = the ruling (a) gates."""
    rows = rows if rows is not None else (record.get('summary') or {})
    ag = record.get('agreement') or {}

    def it(b, c):
        return (rows.get(f'{b}_{c}') or {}).get('iterations')

    out = {}
    a1 = {b: it(b, 'c1_ident_mu') for b in BLOCKS}
    lim = {'tso': PREDICTIONS['A1']['tso_max_iterations'], 'dso': PREDICTIONS['A1']['dso_max_iterations']}
    out['A1'] = {'measured': a1, 'bound': lim,
                 'holds': (all(a1[b] is not None and a1[b] <= lim[b] for b in BLOCKS) if all(a1[b] is not None for b in BLOCKS) else None)}
    pa = PREDICTIONS['A2']['part_a_warm']
    a2 = {b: it(b, 'c2_pert') for b in BLOCKS}
    out['A2'] = {'measured': a2, 'part_a_warm': pa,
                 'ratio': {b: (a2[b] / pa[b] if a2[b] is not None else None) for b in BLOCKS},
                 'holds': (all(abs(a2[b] / pa[b] - 1) <= PREDICTIONS['A2']['rel_band'] for b in BLOCKS)
                           if all(a2[b] is not None for b in BLOCKS) else None)}
    a3 = {b: (it(b, 'c3_pert_mu'), it(b, 'c2_pert')) for b in BLOCKS}
    red = {b: (1 - a3[b][0] / a3[b][1]) if None not in a3[b] else None for b in BLOCKS}
    out['A3'] = {'measured_c3_c2': a3, 'reduction': red,
                 'holds': (all(red[b] >= PREDICTIONS['A3']['min_reduction'] for b in BLOCKS)
                           if all(red[b] is not None for b in BLOCKS) else None)}
    gates = gates if gates is not None else {k: v.get('gate_pass') for k, v in ag.items()}
    resto = {k: v.get('restoration_iterations') for k, v in rows.items()}
    complete = len(gates) == 4 and len(resto) == 6
    out['A4'] = {'gates': gates, 'restoration_iterations': resto,
                 'holds': (all(gates.values()) and all(r == 0 for r in resto.values())) if complete else None}
    w3 = (rows.get('dso_c3_pert_mu') or {}).get('run_smopf_s')
    w2 = (rows.get('dso_c2_pert') or {}).get('run_smopf_s')
    r5 = (1 - w3 / w2) if (w3 is not None and w2) else None
    out['A5'] = {'dso_run_smopf_s_c3': w3, 'dso_run_smopf_s_c2': w2, 'reduction': r5,
                 'holds': (r5 >= PREDICTIONS['A5']['min_reduction']) if r5 is not None else None}
    return out


# ======================================================================================================================
#  run
# ======================================================================================================================
def run(dry_dir=None):
    global _RUN_LOG
    dry = dry_dir is not None
    out_dir = os.path.abspath(dry_dir) if dry else os.path.join(REPO, OUT_REL)
    for forbidden in FORBIDDEN_OUTPUT_ROOTS:
        fa = os.path.abspath(os.path.join(REPO, forbidden))
        if os.path.abspath(out_dir) == fa or os.path.abspath(out_dir).startswith(fa + os.sep):
            raise RuntimeError(f'refusing: output under {forbidden}')
    lock = tp.Lock(os.path.join(REPO, tp.LOCK_REL))   # shared with p516_tperf_profile.py (same output tree)
    lock_info = lock.acquire()
    try:
        others = []
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if proc.info['pid'] not in {os.getpid(), os.getppid()} and 'p516_tperf_a2.py' in cmd and 'python' in cmd:
                others.append({'pid': proc.info['pid'], 'cmdline': cmd[:300]})
        if others:
            raise RuntimeError(f'refusing to run: other p516_tperf_a2.py processes {others}')
        if os.path.exists(out_dir):
            raise RuntimeError(f'refusing: output directory exists: {out_dir}')
        os.makedirs(os.path.join(out_dir, 'logs'))
        os.makedirs(os.path.join(out_dir, 'nl'))
        _RUN_LOG = os.path.join(out_dir, 'run.log')
        record = {'schema': 'p516_tperf_a2b_v1' if CONTINUE else 'p516_tperf_a2_v2',
                  'ruling': RULING_A if CONTINUE else RULING, 'stage': STAGE + (' -- continuation A2b' if CONTINUE else ''),
                  'authority': AUTHORITY + (["Planner ruling (a), 2026-10-31 (continuation after a2's stop)"] if CONTINUE else []),
                  'utc_start': tp._utc(),
                  'mode': ('dry_run' if dry else 'a2') + ('_continuation_a2b' if CONTINUE else ''), 'instance': tp.INSTANCE,
                  'objective_convention': ("the block's active ADMM objective (p58_rescaled_admm_objective) as "
                                           'evaluated by Pyomo at the loaded point -- a local block objective, not '
                                           'gross_operational_cost nor net_operational_recourse; compared only '
                                           'between solves of the same block'),
                  'lock': {**lock_info, 'a2_other_copies_found': others},
                  'thread_env_in_force': {k: os.environ.get(k) for k in tp.THREAD_CAP_ENV},
                  'constants': {'mu_init': MU_INIT, 'perturb_factor': PERTURB_FACTOR, 'obj_rel_gate': OBJ_REL_GATE,
                                'var_abs_gate': VAR_ABS_GATE, 'cases': [list(c) for c in CASES],
                                'target_params': list(TARGET_PARAMS), 'declared_solves_this_launch': DECLARED_SOLVES,
                                'skipped_cases (by reference)': list(SKIPPED_CASES),
                                'var_agree_abs (ruling (a), c3 vs c2)': VAR_AGREE_ABS if CONTINUE else None},
                  'predictions': PREDICTIONS}
        record['provenance'] = {
            'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'hw_model': tp._git_free_cmd(['sysctl', '-n', 'hw.model']),
            'git_head': tp._git(['rev-parse', 'HEAD']), 'git_branch': tp._git(['rev-parse', '--abbrev-ref', 'HEAD']),
            'production_files_git_status': tp._git(['status', '--porcelain', '--'] + list(tp.PRODUCTION_FILES)),
            'production_files_sha256': {f: tp._sha(os.path.join(REPO, f)) for f in tp.PRODUCTION_FILES},
            'harness_sha256': tp._sha(THIS), 'part_a_harness_sha256': tp._sha(tp.THIS),
            'harness_git_status': tp._git(['status', '--porcelain', '--', 'p516_tperf_a2.py', 'p516_tperf_profile.py']),
        }
        if record['provenance']['production_files_git_status']:
            raise RuntimeError(f"production files not clean: {record['provenance']['production_files_git_status']}")
        _log('verifying input hashes (certified_models, esso, production records, post_certification; coupling.json; '
             'Part A baseline)')
        record['inputs'] = tp.verify_inputs()
        record['inputs'].update(verify_side_inputs())
        part_a_stats = {b: final_statistics(os.path.join(REPO, PART_A_WARM_LOG_REL[b])) for b in BLOCKS}
        record['part_a_warm_final_statistics'] = part_a_stats
        if CONTINUE:
            _log('continuation (ruling (a)): verifying a2 by reference (manifest, a2.json, c1 log, harness blob)')
            record['continuation'] = verify_a2_reference(out_dir)
            record['provenance']['part_a_harness_import'] = tp_import_provenance()
            _log(f"  a2 verified: manifest {record['continuation']['a2_manifest_sha256'][:8]}, a2.json "
                 f"{record['continuation']['files_verified'][A2_JSON_REL][:8]}; Part A harness imported "
                 f"{_TP_SHA_AFTER_IMPORT[:8]} (a2 imported b8dbb6aa, committed {TP_COMMITTED_SHA[:8]}); used definitions "
                 f"identical to HEAD: {record['provenance']['part_a_harness_import']['used_definitions_identical_to_head']}")
        # Part A's warm (i) options (hash verified above against Part A's manifest); only the subset ruling (b) needs
        _pa = json.load(open(os.path.join(REPO, PART_A_JSON_REL)))
        part_a_record = {'solves': {f'{b}_i_warm': {k: _pa['solves'][f'{b}_i_warm'][k]
                                                    for k in ('preparation', 'solver_options_passed')} for b in BLOCKS}}
        del _pa
        coupling = json.load(open(os.path.join(REPO, COUPLING_REL)))
        mags = channel_magnitudes(coupling)
        del coupling
        record['perturbation'] = {'label': 'perturbation = 10 x terminal mismatch',
                                  'source': f'{COUPLING_REL} mismatch_step1[{T01B_CELL}] (read, not recomputed)',
                                  'definition': ('shift = +PERTURB_FACTOR x max |mismatch| over the block 2025 Spring '
                                                 'entries of the channel, uniform over every index of the target Param'),
                                  'magnitudes': mags,
                                  'deltas': {b: perturbation_deltas(mags[b]) for b in BLOCKS}}
        for b in BLOCKS:
            for pname, d in record['perturbation']['deltas'][b].items():
                _log(f'  perturbation {b} {pname}: +{d["shift_physical"]:.6g} {d["unit_physical"]} '
                     f'= +{d["shift_param_units"]:.6g} {d["param_units"]}')
        record['certification_options'] = certification_options()

        import pyomo
        import pyomo.environ as pe
        import network as network_module
        record['provenance']['pyomo_version'] = pyomo.version.version

        _log('unpickling certified_models.pkl')
        with open(os.path.join(REPO, tp.INPUTS['certified_models'][0]), 'rb') as handle:
            payload = pickle.load(handle)
        originals = {'tso': payload['tso'][tp.YEAR][tp.DAY], 'dso': payload['dso'][tp.DSO_NODE][tp.YEAR][tp.DAY]}
        certified = {}
        prepared = {}
        record['blocks'] = {}
        for block in BLOCKS:
            orig = originals[block]
            net0, prm0 = tp._build_time_objects(orig)
            if net0.name != NETWORK_NAME[block]:
                raise RuntimeError(f'{block}: network {net0.name}')
            if net0.baseMVA != mags[block]['s_base_MVA']:
                raise RuntimeError(f'{block}: baseMVA {net0.baseMVA} != coupling s_base {mags[block]["s_base_MVA"]}')
            vb_net = {}
            for n in (BLOCK_NODES[block] if block == 'tso' else (mags[block]['constants'][str(tp.DSO_NODE)]['dso_ref_node'],)):
                if net0.node_exists(n):
                    vb_net[str(n)] = net0.get_node_base_kv(n)
            if not vb_net or set(vb_net.values()) != {mags[block]['v_base_kV']}:
                raise RuntimeError(f'{block}: node base kV {vb_net} != coupling v_base {mags[block]["v_base_kV"]}')
            certified[block] = primal_snapshot(orig, pe)
            record['blocks'][block] = {
                'model_name': orig.name, 'network': net0.name, 'year': net0.year, 'day': net0.day,
                'baseMVA': net0.baseMVA, 'node_base_kV_checked': vb_net,
                'pickled_production_options': dict(prm0.solver_params.options or {}),
                'mu_init_in_pickled_options': 'mu_init' in (prm0.solver_params.options or {}),
                'certified_objective': certified[block]['objective'],
                'certified_objective_name': certified[block]['objective_name'],
                'n_free_vars': len(certified[block]['vars']),
                'target_params_present': {p: hasattr(orig, p) for p in TARGET_PARAMS},
                'suffix_sizes': {s: len(getattr(orig, s)) for s in ('dual', 'ipopt_zL_out', 'ipopt_zU_out')}}
            if not all(record['blocks'][block]['target_params_present'].values()):
                raise RuntimeError(f'{block}: target Params missing')
            if record['blocks'][block]['mu_init_in_pickled_options']:
                raise RuntimeError(f'{block}: pickled options already carry mu_init')
            for case, perturbed, mu in CASES:
                ck = f'{block}_{case}'
                model = orig.clone()
                net, prm = tp._build_time_objects(model)
                if net is net0 or prm is prm0:
                    raise RuntimeError('clone shares network/params with the loaded block')
                net.logs_dir = os.path.join(out_dir, 'logs')
                net.ipopt_solve_records = deque(maxlen=network_module.IPOPT_SOLVE_RECORDS_MAXLEN)
                production_options = dict(prm.solver_params.options or {})
                options = dict(production_options)
                options['output_file'] = f'a2_{ck}.log'
                options.update(tp.ALL_CASE_OPTIONS)
                if mu:
                    options['mu_init'] = MU_INIT
                prm.solver_params.options = options
                entry = {'options_added_or_changed': {k: v for k, v in options.items()
                                                      if production_options.get(k) != v},
                         'production_options': production_options, 'perturbed': perturbed, 'mu_init': mu}
                if perturbed:
                    entry['perturbation_applied'] = apply_perturbation(
                        model, record['perturbation']['deltas'][block], network_module, pe)
                tp.CASE_IS_WARM[case] = True
                prepared[ck] = {'block': block, 'case': case, 'model': model, 'network': net, 'params': prm,
                                'entry': entry}
        record['comparator_check'] = comparator_check(
            {b: certified[b]['objective'] for b in BLOCKS}, record['certification_options'],
            {b: prepared[f'{b}_c1_ident_mu']['params'].solver_params.options for b in BLOCKS}, part_a_record)
        for b, row in record['comparator_check']['blocks'].items():
            rr = row['recorded_not_gated']
            _log(f'  comparator {b}: c1 comparator (Part A warm limit) = '
                 f'{row["c1_objective_comparator (Part A warm limit, log final unscaled)"]!r} '
                 f'(== declared {row["declared_by_ruling"]!r}: {row["parsed_equals_declared_exactly"]}); option regime '
                 f'matches Part A warm: {row["option_regime_matches_part_a_warm"]} {row["option_regime_differences"]}; '
                 f'recorded: certified f = {rr["certified_objective (pe.value at the pickled point)"]!r}, limit - certified '
                 f'= {rr["part_a_limit_minus_certified"]:.6g} ({rr["part_a_limit_rel_diff"]:.3e} rel), compl_inf_tol '
                 f'passed {rr["compl_inf_tol_passed_this_run"]} vs at certification '
                 f'{rr["compl_inf_tol_passed_at_certification"]}')
        _log(f'comparator check: {record["comparator_check"]["verdict"]}')
        del originals
        del payload
        gc.collect()

        inst = tp.Instrument(os.path.join(out_dir, 'nl'), dry).install()
        # A2b: TSO c1 is prepared (its options enter the comparator check) but NOT run (by reference to a2)
        to_run = {ck: p for ck, p in prepared.items() if ck not in SKIPPED_CASES}
        if CONTINUE and sorted(to_run) != sorted(SPLIT['a2b']['solves']):
            raise RuntimeError(f'cases to run {sorted(to_run)} != ruling (a) {sorted(SPLIT["a2b"]["solves"])}')
        if len(to_run) != DECLARED_SOLVES:
            raise RuntimeError(f'{len(to_run)} cases to run != declared {DECLARED_SOLVES}')
        record['capture_checklist'] = tp.capture_checklist(inst, to_run, network_module, out_dir, dry)
        own = {}
        for ck, p in to_run.items():
            opts = p['params'].solver_params.options
            own[ck] = {
                'mu_init_iff_mu_case': (opts.get('mu_init') == MU_INIT) if CASE_MU[p['case']] else ('mu_init' not in opts),
                'perturbation_iff_perturbed_case': (('perturbation_applied' in p['entry']) == CASE_PERTURBED[p['case']]),
                'iteration_table_and_mu_lines_logged (file_print_level >= 5)': int(opts.get('file_print_level', 5)) >= 5,
                'mu_trace_parser_selftest_on_part_a_log': None,
            }
        st = mu_trace(open(os.path.join(REPO, PART_A_WARM_LOG_REL['tso'])).read())
        st_ok = st['lg_mu_iter0_table'] == -1.0 and st['mu_first_current_barrier_line'] == 0.1
        for c in own.values():
            c['mu_trace_parser_selftest_on_part_a_log'] = st_ok
        record['capture_checklist']['a2_items'] = own
        record['capture_checklist']['a2_mu_trace_selftest'] = st
        fails = [f'{ck}:{k}' for ck, c in own.items() for k, v in c.items() if v is not True]
        if fails:
            raise RuntimeError(f'A2 capture checklist failed before any solve: {fails}')
        if record['comparator_check']['failures'] and not dry:
            record['status'] = 'refused_before_any_solve'
            record['stopped'] = f'comparator check (ruling (b)) failed: {record["comparator_check"]["failures"]}'
            record['guard'] = {'counts': dict(_GUARD.counts), 'declared_solves': DECLARED_SOLVES,
                               'verify_failures': 'not verified: refused before any solve'}
            record['utc_end'] = tp._utc()
            tp._write_json(os.path.join(out_dir, 'a2.json'), record)
            _log(f'REFUSED before any solve: {record["stopped"]}; guard {_GUARD.counts}')
            return 3
        pred_path = os.path.join(out_dir, 'predictions.json')
        tp._write_json(pred_path, {'utc': tp._utc(), 'predictions': PREDICTIONS,
                                   'perturbation_deltas': record['perturbation']['deltas'],
                                   'harness_sha256': record['provenance']['harness_sha256'],
                                   **({'ruling_a': RULING_A, 'split': SPLIT,
                                       'note': ('predictions unchanged from a2 (recorded before a2, a2/predictions.json '
                                                'ca4b850d...); gates restated by ruling (a) before any a2b solve')}
                                      if CONTINUE else {})})
        record['predictions_file'] = {'path': os.path.relpath(pred_path, REPO), 'sha256': tp._sha(pred_path),
                                      'written_before_first_solve': True}
        _log('capture checklist holds; predictions written; starting the declared solves'
             + (' (DRY: all blocked)' if dry else ''))

        record['solves'] = {}
        record['agreement'] = {}
        solutions = {}
        stats = {}
        stopped = None
        if CONTINUE:
            ref = record['continuation']
            ag_ref = ref['tso_c1_agreement_in_a2 (ruling (b) form)']
            record['agreement']['tso_c1_vs_part_a_limit (a2 solve, by reference)'] = {
                'source': 'a2 (solve not re-run); restated gate evaluated in a2b from a2.json + the a2 c1 log (zero solves)',
                'slack_gate': slack_gate('tso c1 (a2) vs Part A TSO warm limit', ag_ref['objective_a'],
                                         PART_A_WARM_LIMIT_DECLARED['tso'], ref['tso_c1_final_statistics'],
                                         part_a_stats['tso'], 'tso_c1_ident_mu (a2)', 'part_a_tso_i_warm'),
                'ipopt_log_objective_c1': ref['tso_c1_final_statistics']['objective']['unscaled'],
                'ruling_b_form_in_a2 (superseded)': {k: ag_ref[k] for k in ('objective_rel_diff', 'gate_pass')}}
            g = record['agreement']['tso_c1_vs_part_a_limit (a2 solve, by reference)']['slack_gate']
            _log(f"  restated gate tso c1 (a2, by reference): |df| {g['abs_delta_f']:.4e} / bar "
                 f"{g['bar (max terminal unscaled complementarity)']:.4e} = {g['ratio_abs_delta_f_over_bar']:.4f} -> "
                 f"{g['verdict']}")
        for ck, p in to_run.items():
            block, case = p['block'], p['case']
            _log(f'case {ck}: run_smopf(from_warm_start=True) perturbed={CASE_PERTURBED[case]} mu_init='
                 f'{p["params"].solver_params.options.get("mu_init")}')
            rec, ok = tp.run_case(ck, block, case, p['model'], p['network'], p['params'], inst, network_module, pe,
                                  os.path.join(out_dir, 'logs'), dry)
            rec['preparation'] = p['entry']
            log_path = os.path.join(REPO, rec['ipopt_output_file'])
            rec['mu_trace'] = mu_trace(open(log_path).read()) if os.path.exists(log_path) else None
            record['solves'][ck] = rec
            row = solve_row(rec)
            _log(f'case {ck}: ok={ok} term={row["termination"]} iters={row["iterations"]} '
                 f'restoration={row["restoration_iterations"]} mu0={row["mu_iter0_table"]} mu_final={row["mu_final"]} '
                 f'run_smopf={row["run_smopf_s"]:.3f}s ipopt={row["ipopt_process_s"]}')
            if dry:
                continue
            if not ok or rec['solve_calls_in_case'] != 1 or rec['cap_blocked_in_case']:
                stopped = (f'solve failed or count off at {ck}: ok={ok} calls={rec["solve_calls_in_case"]} '
                           f'cap_blocked={rec["cap_blocked_in_case"]} -- stopping (task: record and stop)')
            elif row['restoration_iterations'] is None or row['restoration_iterations'] > 0:
                stopped = f'restoration at {ck} ({row["restoration_iterations"]} lines) -- stopping (task: record and stop)'
            if stopped:
                _log(stopped)
                break
            solutions[ck] = primal_snapshot(p['model'], pe)
            if CONTINUE:
                stats[ck] = final_statistics(log_path)
                rec['final_statistics'] = stats[ck]
            if case == 'c1_ident_mu':
                ag = compare_c1(solutions[ck], block, certified[block])
                f_log = log_final_objective(log_path)
                ag['ipopt_log_final_objective_c1'] = f_log
                ag['ipopt_log_vs_part_a_log_rel_diff (reported)'] = (
                    abs(f_log - PART_A_WARM_LIMIT_DECLARED[block]) / abs(PART_A_WARM_LIMIT_DECLARED[block])
                    if f_log is not None else None)
                pa_passed = _regime(part_a_record['solves'][f'{block}_i_warm']['solver_options_passed'], ('output_file',))
                a2_passed = _regime(rec['solver_options_passed'], OPTION_KEYS_EXCLUDED_FROM_REGIME_MATCH)
                ag['options_passed_to_solver_match_part_a_warm (post-solve, reported)'] = bool(pa_passed == a2_passed)
                ag['options_passed_differences'] = {k: [pa_passed.get(k), a2_passed.get(k)]
                                                    for k in sorted(set(pa_passed) | set(a2_passed))
                                                    if pa_passed.get(k) != a2_passed.get(k)}
                if CONTINUE:
                    ag['ruling_b_gate (1e-9 relative; reported beside, not gated under ruling (a))'] = {
                        k: ag.pop(k) for k in ('objective_within_gate', 'gate_pass')}
                    ag['slack_gate'] = slack_gate(f'{block} c1 vs Part A {block.upper()} warm limit', ag['objective_a'],
                                                  PART_A_WARM_LIMIT_DECLARED[block], stats[ck], part_a_stats[block],
                                                  ck, f'part_a_{block}_i_warm')
                    ag['slack_gate_ipopt_log_objective (reported)'] = slack_gate(
                        f'{block} c1 vs Part A (IPOPT log objectives)', stats[ck]['objective']['unscaled'],
                        part_a_stats[block]['objective']['unscaled'], stats[ck], part_a_stats[block], ck,
                        f'part_a_{block}_i_warm')
                    ag['gate_pass'] = ag['slack_gate']['pass']
                record['agreement'][f'{block}_c1_vs_part_a_limit'] = ag
                vc = ag['vars_vs_certified_reported_not_gated']
                _log(f'  gate {block} c1 objective vs Part A warm limit: f = {ag["objective_a"]!r} vs '
                     f'{PART_A_WARM_LIMIT_DECLARED[block]!r}, rel {ag["objective_rel_diff"]:.3e} -> pass={ag["gate_pass"]}; '
                     f'IPOPT log f = {f_log!r}; options match Part A: '
                     f'{ag["options_passed_to_solver_match_part_a_warm (post-solve, reported)"]}; REPORTED (not gated) '
                     f'vs certified: obj rel {vc["objective_rel_diff"]:.3e}, max |dx| {vc["max_abs_var_diff"]:.3e} p.u., '
                     f'{vc["n_vars_over_gate"]} Vars > 1e-6')
                if CONTINUE:
                    g = ag['slack_gate']
                    _log(f"  restated gate {block} c1: |df| {g['abs_delta_f']:.4e} / bar "
                         f"{g['bar (max terminal unscaled complementarity)']:.4e} = {g['ratio_abs_delta_f_over_bar']:.4f} "
                         f"-> {g['verdict']} (1e-9 figure: {g['recorded_not_gated']['objective_rel_diff']:.3e} rel, "
                         'reported beside)')
                elif not ag['gate_pass']:
                    stopped = (f"c1 objective gate (vs Part A's warm limit, ruling (b)) failed on {ck} -- stopping "
                               '(task: report, do not tune)')
                    _log(stopped)
                    break
            if case == 'c3_pert_mu':
                ag = compare(solutions[ck], solutions[f'{block}_c2_pert'], f'{ck} vs {block}_c2_pert')
                if CONTINUE:
                    ag['ruling_b_gate (1e-9 objective, 1e-6 every Var; reported beside, not gated under ruling (a))'] = {
                        k: ag.pop(k) for k in ('objective_within_gate', 'vars_within_gate', 'gate_pass')}
                    c2k = f'{block}_c2_pert'
                    ag['slack_gate'] = slack_gate(f'{block} c3 vs c2', ag['objective_a'], ag['objective_b'],
                                                  stats[ck], stats[c2k], ck, c2k)
                    ag['slack_gate_ipopt_log_objective (reported)'] = slack_gate(
                        f'{block} c3 vs c2 (IPOPT log objectives)', stats[ck]['objective']['unscaled'],
                        stats[c2k]['objective']['unscaled'], stats[ck], stats[c2k], ck, c2k)
                    mx = ag['max_abs_var_diff']
                    ag['var_agreement_ruling_a'] = {
                        'max_abs_dx_pu': mx, 'n_vars_over_1e-6': ag['n_vars_over_gate'], 'bound_pu': VAR_AGREE_ABS,
                        'n_missing': ag['n_missing'], 'n_value_none': ag['n_value_none'],
                        'agree': bool(mx is not None and mx <= VAR_AGREE_ABS),
                        'rule': 'agreement iff max |dx| <= 1e-4 p.u.; otherwise reported, not a stop'}
                    ag['gate_pass'] = bool(ag['slack_gate']['pass'] and ag['var_agreement_ruling_a']['agree'])
                    g = ag['slack_gate']
                    _log(f"  restated gate {block} c3 vs c2: |df| {g['abs_delta_f']:.4e} / bar "
                         f"{g['bar (max terminal unscaled complementarity)']:.4e} = {g['ratio_abs_delta_f_over_bar']:.4f} "
                         f"-> {g['verdict']}; max |dx| {mx:.3e} p.u. (agree <= 1e-4: "
                         f"{ag['var_agreement_ruling_a']['agree']}), {ag['n_vars_over_gate']} Vars > 1e-6")
                record['agreement'][f'{block}_c3_vs_c2'] = ag
                if not CONTINUE:
                    _log(f'  gate {block} c3 vs c2: obj rel {ag["objective_rel_diff"]:.3e}; max |dx| '
                         f'{ag["max_abs_var_diff"]:.3e}; over gate {ag["n_vars_over_gate"]} -> pass={ag["gate_pass"]}')
        record['stopped'] = stopped

        expected = 0 if dry else DECLARED_SOLVES
        if dry:
            gfails = []
            if _GUARD.counts['permitted_solve'] or _GUARD.counts['permitted_exec']:
                gfails.append(f'dry run reached a permitted solve {_GUARD.counts}')
            if _GUARD.counts['blocked_solve'] != len(record['solves']):
                gfails.append(f"blocked {_GUARD.counts['blocked_solve']} != cases {len(record['solves'])}")
        else:
            gfails = _GUARD.verify(expected)
        record['guard'] = {'permitted': [list(x) for x in PERMITTED] if not dry else [],
                           'declared_solves': expected, 'counts': dict(_GUARD.counts),
                           'verify_failures': gfails, 'per_case_cap_blocks': inst.cap_blocks}
        record['status'] = ('complete' if not gfails and stopped is None else ('stopped' if stopped else 'guard_failed'))
        record['summary'] = {ck: solve_row(rec) for ck, rec in record['solves'].items()} if not dry else None
        if not dry and CONTINUE:
            rows6 = {'tso_c1_ident_mu': {**record['continuation']['tso_c1_summary_row'], 'source': 'a2 (by reference)'}}
            rows6.update({k: {**v, 'source': 'a2b'} for k, v in record['summary'].items()})
            record['summary_all_six'] = rows6
            gates_a = {k: v.get('slack_gate', {}).get('pass') if 'c1' in k else v.get('gate_pass')
                       for k, v in record['agreement'].items()}
            gates_a = {('tso_c1_vs_part_a_limit' if k.startswith('tso_c1_vs_part_a_limit') else k): v
                       for k, v in gates_a.items()}
            try:
                record['predictions_outcome'] = evaluate_predictions(record, rows=rows6, gates=gates_a)
                record['predictions_outcome']['A4']['gate_form'] = (
                    'ruling (a): objective = stopping-slack ratio < 1 on all four; c3 vs c2 also max |dx| <= 1e-4 p.u.')
                orig = {}
                for k, v in record['agreement'].items():
                    if 'c1' in k:
                        sg = v['slack_gate']
                        orig[k] = bool(sg['recorded_not_gated']['objective_rel_diff'] <= OBJ_REL_GATE)
                    else:
                        orig[k] = bool(v['objective_rel_diff'] <= OBJ_REL_GATE and v['n_vars_over_gate'] == 0
                                       and v['n_missing'] == 0 and v['n_value_none'] == 0)
                record['predictions_outcome']['A4_original_form (1e-9 objective, 1e-6 every Var; reported)'] = {
                    'gates': orig, 'holds': all(orig.values()) and all(
                        r.get('restoration_iterations') == 0 for r in rows6.values())}
            except Exception as error:  # noqa: BLE001
                record['predictions_outcome'] = {'error': f'{type(error).__name__}: {error}',
                                                 'traceback': traceback.format_exc()}
        elif not dry:
            try:
                record['predictions_outcome'] = evaluate_predictions(record)
            except Exception as error:  # noqa: BLE001
                record['predictions_outcome'] = {'error': f'{type(error).__name__}: {error}',
                                                 'traceback': traceback.format_exc()}
        record['utc_end'] = tp._utc()
        record['wall_s'] = time.time() - T0
        name = f'{_LOG_STEM}.json' if not dry else f'{_LOG_STEM}_dry_run.json'
        tp._write_json(os.path.join(out_dir, name), record)
        _log(f'status={record["status"]} guard={_GUARD.counts} verify_failures={gfails} -> {name}')
        return 0 if record['status'] == 'complete' else 2
    finally:
        lock.release()


def manifest():
    root = os.path.join(REPO, OUT_REL)
    target = os.path.join(root, 'manifest_sha256.json')
    if os.path.exists(target):
        raise RuntimeError(f'refusing to overwrite {target}')
    files = {}
    for dirpath, _, names in os.walk(root):
        for nm in sorted(names):
            p = os.path.join(dirpath, nm)
            files[os.path.relpath(p, REPO)] = tp._sha(p)
    for rel in LAUNCH_LOGS_REL:
        if os.path.exists(os.path.join(REPO, rel)):
            files[rel] = tp._sha(os.path.join(REPO, rel))
    for p in (THIS, tp.THIS):
        files[os.path.relpath(p, REPO)] = tp._sha(p)
    for rel, _ in tp.INPUTS.values():
        files[rel] = tp._sha(os.path.join(REPO, rel))
    for rel in [COUPLING_REL, T01B_MANIFEST_REL, PART_A_MANIFEST_REL, PART_A_JSON_REL] + list(PART_A_WARM_LOG_REL.values()):
        files[rel] = tp._sha(os.path.join(REPO, rel))
    extra = {}
    if CONTINUE:
        for rel in (A2_MANIFEST_REL, A2_JSON_REL, A2_C1_LOG_REL):
            files[rel] = tp._sha(os.path.join(REPO, rel))
        extra = {'split': SPLIT, 'note': ('p516_tperf_profile.py hashed at manifest time; the bytes a2b imported are '
                                          'recorded in a2b.json provenance.part_a_harness_import')}
    tp._write_json(target, {'schema': 'p516_tperf_a2b_manifest_v1' if CONTINUE else 'p516_tperf_a2_manifest_v1',
                            'utc': tp._utc(), 'files': files, **extra})
    print(f'manifest: {len(files)} files -> {target}')
    return 0


if __name__ == '__main__':
    try:
        if ARGS.manifest:
            sys.exit(manifest())
        sys.exit(run(ARGS.dry_run))
    except Exception:
        traceback.print_exc()
        if _GUARD is not None:
            print(f'guard counts at error: {_GUARD.counts}', file=sys.stderr, flush=True)
        sys.exit(1)
