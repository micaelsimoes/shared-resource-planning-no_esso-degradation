"""
P5.15 W183i (c) -- the ZERO-SOLVE gate of the E1 launcher v3: per-poll / per-batch RESUMABLE launches
(PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 3 (iii)-(v)).

WHAT IS TESTED (stub records only; no solve, no pickle, no model build).
  The launcher's own launch cores `search_launch` / `seeds_launch` (the code --run --resumable runs) with ONE
  replacement: the harness `H.evaluate(labels, ctx, heartbeat_tag)` is a deterministic STUB HARNESS (`StubHarness`)
  that writes, for every evaluation, the files H.evaluate and its child write and the launcher reads
  (evaluation_record.json, per_cycle_record.jsonl, exit_code.txt, wait4_rusage.json, the heartbeat file) and returns
  the records with `parent_view`, as H.evaluate does. Every planted E1 root lives in --scratch (outside the
  repository); each is built by the launcher's own seeds launches and a search spec frozen by the harness's own
  `freeze_campaign_spec` (the search-phase `extra` as `freeze` writes it: arm, cache_table, initial_incumbent).

  1. SPLIT == IN-PROCESS, byte-equal (the W183g (b) method: the same writer `gate_result_io.dumps(sort_keys=True)`,
     the SAME stub object, a deep-copied cache). The reference is the in-process `run_continuation_e1` loop called
     ONCE, with the v2 --run evaluate_fn logic (labels from keys, `batch_NNN` heartbeat tags, `point_from_harness`,
     `cache_entry_from_point`) over the stub harness, in the same root path. The split runs are `search_launch`
     called once per launch (each call a fresh rebuild from disk; nothing carried in memory) with the launch unit
     `batch` (a split at EVERY batch boundary) and `poll` (at every evaluating-poll boundary). Compared: the return
     dict, the poll history (concatenated launch records), the points, the stub calls, every file under evals/,
     the heartbeat names. Scenarios: an advanced (B+) poll followed by further evaluating polls (a split right after
     a B+ advance, and inside the advanced poll), A7 per poll (arm N, x = 0 through the G0 alias), A7 overall, a move
     to a storage incumbent whose next poll advances, budget exhaustion at concurrency 4; after the terminating
     launch (A7 stop included) a further launch is refused. The seeds phase split in batches of its own size
     (concurrency 1: seven launches; 4: two; the batch root's four seeds at 4: one) against one unsplit call.
  2. FOREIGN RECORDS refused (planted in a mid-run phase root, then restored): a point whose eval dir lies in
     another root; a wrong eval key under the arm's model; a record whose bytes no longer match the manifest; a
     record of the other arm (as is, and relabelled under this arm's label); an unlisted eval dir; a launch record
     of another root. Each refusal names the failing check; nothing is evaluated (0 stub calls) and nothing written.
  3. DISAGREEMENT refused: the last launch record's state tampered (k_next, new-evaluation count, barrier count,
     incumbent, poll in progress) with the manifest re-hashed, and a persisted poll record tampered; the seeds
     phase's evaluated count tampered. Then the untampered root resumes and completes byte-equal (positive control).
  4. THE W183g (b) GATE CHECKS ON v3: advance off == imported `C.run_continuation` (8 seed cases, bytes); advance on
     -> k = 2/2/1/0 with sizes 8/9/8/14 and skipped [0, 1]/[0, 1]/[0]/[] (W183b's table, pinned); the k = 0..9 table
     reproduced; the item-3 paths (A7 per poll / overall, move to storage, budget 0 / 5, max_polls 1 x 2, window
     exhaustion (iii) at the production window on W183g (b)'s committed first hit); a probe passed changes nothing
     (8 cases, bytes); the source diff of `run_continuation_e1` against `C.run_continuation`, every changed line in
     {signature, k_next state, B+ block, resume hook}.
  5. THE v1/v2 DRY-RUN ASSERTIONS ON v3: `L.dry_run` in-process (7 keys + 1 G0 alias with ca195bd3..., base keys rows
     35 / 40, root-scoped cache, pins, preflight, rule eleven incl. the resume-hook capture), g0_check with
     require_current_production, the declared roots / batch seeds; plus the CLI flag rules and the 'resume' root rule.
  The per-launch time bound is computed from the on-disk evaluation records (JSON reads only).

GUARDS. SolveProfileGuard(permitted=()) installed before any other project import and verified at exactly 0 with
every guard the imports install; pickle.load / loads blocked (the launcher's counters, 0); the launcher's pyomo
ConcreteModel tripwire and its 52 tripwires on the committed launchers' functions at 0.

EXACT COMMAND (repo root, canonical interpreter, attached, alone, both streams captured):
  mkdir data/SRP1/Results/P515S55/w183i_c_e1_v3_resume && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183i_c_e1_v3_resume_gate.py \\
      --out data/SRP1/Results/P515S55/w183i_c_e1_v3_resume \\
      --g0-artifact data/SRP1/Results/P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz \\
      --g0-sha256 ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a \\
      --scratch <a new directory outside the repository> \\
      > data/SRP1/Results/P515S55/w183i_c_e1_v3_resume/launch.log 2>&1
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GATE_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183i (c) E1 v3 resume gate (never solves)').install()

import argparse  # noqa: E402
import copy  # noqa: E402
import difflib  # noqa: E402
import glob  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import p515_s55_e1_launcher as L  # noqa: E402  (installs its guard, blocks pickle, the model-build and tripwires)
import p515_s55_w183g_b_e1_v2_dryrun as G  # noqa: E402  (the W183g (b) gate: stub, paired runs, comparators; main() NOT run)

C, PB, H, GRIO = L.C, L.PB, L.H, L.GRIO

SCRIPT_NAME = os.path.basename(__file__)
OWN_SUBSTRINGS = ('p515_s55_w183i_c_e1_v3_resume_gate', 'p515_s55_w183g_b_e1_v2_dryrun', 'p515_s55_e1_launcher')
OUT_FILE = 'w183i_c_e1_v3_resume_gate.json'
OUT_RUNS_FILE = 'w183i_c_loop_outputs.json.gz'
OUT_MANIFEST = 'w183i_c_e1_v3_resume_gate_manifest_sha256.json'
LAUNCHER_DRY_RUN_SUBDIR = 'launcher_dry_run'
LAUNCHER_V2_SHA256 = 'b0841b090c2c7deaf065c68f35d6bd2bf05e5fc9ef9faa966200ad3c8d0d8773'
W183G_B_GATE = {'path': 'p515_s55_w183g_b_e1_v2_dryrun.py',
                'sha256': '72905f64f2370f1414e3f273357064a056c79f9ee7f300709ffc6913ae5186ad'}
W183G_B_OUT = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183g_b_e1_v2_dryrun',
                                    'w183g_b_e1_v2_dryrun.json'),
               'sha256': '9c1eb3da2be3269437f3ed2b98807b1fd9bc0a6252cf6a7562fb8d25f5e3d7ca'}
UNCHANGED_MODULES = ('p515_s53_f2_certificate.py', 'p515_s47_phase_b_record.py', 'p515_s44_campaign_harness.py',
                     'p515_s55_step7_resettle_hooks.py')
HARNESS_SHA256 = 'd9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191'   # W183i (a), router patch
Q0 = G.Q0
STUB_BAR = G.STUB_BAR
MAX_LAUNCHES = 400


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W183i-c] {msg}', flush=True)


def own_copies_alive():
    """Another copy of this gate, of the W183g (b) gate or of the launcher (the process tree of this run excluded)."""
    import subprocess
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and 'python' in parts[1] \
                and any(x in parts[1] for x in OWN_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def _sha(rel):
    return H.sha256_file(rel if os.path.isabs(rel) else os.path.join(REPO, rel))


_ser = L._ser


def _rel(path):
    return os.path.relpath(path, REPO)


# ======================================================================================================================
#  the stub harness (the only replacement): H.evaluate's file contract, deterministic, zero solves
# ======================================================================================================================
class StubHarness:
    """Stand-in for `H.evaluate(batch, ctx, heartbeat_tag)`. Resolves the labels in ctx.spec (H._spec_candidate),
    refuses an existing eval dir or tagged heartbeat file (write-once, as H.evaluate), writes per evaluation the files
    the launcher reads -- evaluation_record.json (every field point_from_harness / point_problems read; Q = q_of(label),
    status certified unless the label is a barrier), per_cycle_record.jsonl, exit_code.txt, wait4_rusage.json -- then
    the heartbeat file, and returns the records plus `parent_view`, as H.evaluate. No clock, no pid: two runs in the
    same root path write byte-identical files. Every call is logged."""

    def __init__(self, q_of, barrier=(), name='stub'):
        self.q_of, self.barrier, self.name = q_of, set(barrier), name
        self.calls = []
        self.last_batch_info = {}

    def record(self, entry, eval_dir, spec):
        lab = entry['label']
        bad = lab in self.barrier
        q = None if bad else float(self.q_of(lab))
        rec = {'schema': 'w183i_c_stub_evaluation_record', 'stub': self.name, 'campaign_id': spec['campaign_id'],
               'candidate_label': lab, 'candidate_canonical': entry['canonical'], 'candidate_key': entry['key'],
               'eval_key': entry['eval_key'], 'eval_dir': os.path.relpath(eval_dir, REPO),
               'status': 'not_certified' if bad else 'certified', 'certified_cost': q,
               'terminal_gross_operational_cost': q, 'terminal_net_operational_recourse': None if bad else q - 1.0,
               'bar': {'value': None if bad else STUB_BAR}, 'barrier_cause': 'w183i_c_stub_barrier' if bad else None,
               'cycles_run': 500 if bad else 120, 'certification_cycle': None if bad else 120,
               'rule_ten': {'terminal_step_over_threshold': 0.5},
               'flex_price_multiplier': entry.get('flex_price_multiplier'),
               'flex_price_label': entry.get('flex_price_label'),
               'flex_price_applied_in_child': {'readback_pre_run': {'all_match': True}},
               'flex_price_readback_terminal': {'all_match': True}, 'storage_per_node': None,
               'wall_time_s': {'child_process_s': 1.0}, 'peak_rss': 1}
        if entry.get('model_variant') is not None:
            rec.update({'model_variant': entry['model_variant'], 'model_variant_label': entry.get('model_variant_label'),
                        'model_variant_readback_pre_run': {'all_match': True},
                        'model_variant_readback_terminal': {'all_match': True}})
        else:
            rec.update({'ess_ageing_verified_pre_run': {'readback_pre_run': {'all_match': True}},
                        'ess_ageing_readback_terminal': {'all_match': True}})
        return rec

    def __call__(self, batch, ctx, heartbeat_tag=None):
        entries = [H._spec_candidate(ctx, x) for x in batch]
        self.calls.append({'labels': [e['label'] for e in entries], 'heartbeat_tag': heartbeat_tag,
                           'phase_root': os.path.basename(ctx.campaign_root)})
        hb = os.path.join(ctx.campaign_root, H.heartbeat_file_name(heartbeat_tag))
        if heartbeat_tag is not None and os.path.exists(hb):
            raise RuntimeError(f'stub harness: heartbeat exists (write-once per tag): {hb}')
        os.makedirs(ctx.evals_root, exist_ok=True)
        for e in entries:
            if os.path.exists(os.path.join(ctx.evals_root, e['eval_dir'])):
                raise RuntimeError(f"stub harness: evaluation dir already exists (write-once): {e['eval_dir']}")
        out = []
        for e in entries:
            ed = os.path.join(ctx.evals_root, e['eval_dir'])
            os.makedirs(ed)
            rec = self.record(e, ed, ctx.spec)
            with open(os.path.join(ed, 'evaluation_record.json'), 'x') as handle:
                handle.write(GRIO.dumps(rec, indent=1, sort_keys=True))
            with open(os.path.join(ed, 'per_cycle_record.jsonl'), 'x') as handle:
                handle.write(json.dumps({'cycle': 1, 'stub': self.name, 'label': e['label']}, sort_keys=True) + '\n')
            with open(os.path.join(ed, 'exit_code.txt'), 'x') as handle:
                handle.write('0\n')
            with open(os.path.join(ed, 'wait4_rusage.json'), 'x') as handle:
                handle.write(json.dumps({'rusage': {'ru_maxrss': 1}, 'wall_s_parent_view': 1.0, 'exit_code': 0,
                                         'semantics': 'w183i_c stub'}, indent=1, sort_keys=True))
            out.append(dict(rec, parent_view={'exit_code': 0, 'wall_s': 1.0, 'wait4_ru_maxrss': 1}))
        with open(hb, 'w') as handle:
            handle.write(json.dumps({'stub': self.name, 'heartbeat_tag': heartbeat_tag,
                                     'done': [e['label'] for e in entries]}, indent=1, sort_keys=True))
        self.last_batch_info = {'max_concurrent_observed': min(len(entries), int(ctx.concurrency)),
                                'stub': self.name}
        return out


def lattice_label_of(label):
    """'y2025__n7_p0.5_e2_A_2x' -> 'y2025__n7_p0.5_e2' (the domain label minus the model/price suffix)."""
    for m in L.MODELS:
        suffix = f'_{m}_{L.FLEX_M:g}x'
        if label.endswith(suffix):
            return label[:-len(suffix)]
    return label


# ======================================================================================================================
#  planted E1 roots (scratch): seeds by the launcher's own seeds launches, the search spec by the harness's freeze
# ======================================================================================================================
G0_PLANTED = None   # (accepted, info) set in main: the final-head G0 sha, as the cache loader checks it


def plant_seeds_spec(root_abs, root_rel, lattice, labels, concurrency, g0_sha):
    plan = L.seed_plan(lattice)
    entries, rows = L.seeds_entries(lattice, plan, set(labels))
    needs_alias = any(r['seed'] == L.ALIAS_SEED and r['model'] == L.ALIAS_TARGET_MODEL for r in rows)
    alias = L.alias_record(plan, 'planted_g0', {'sha256_on_disk': g0_sha, 'accepted': True})
    extra = {'campaign_script': L.SCRIPT_NAME, 'e1_root': root_rel, 'phase': 'seeds',
             'aliases': [alias] if needs_alias else [], 'w183i_c_planted': True}
    croot = L.phase_root(root_abs, root_rel, 'seeds')
    path, sha, spec = H.freeze_campaign_spec(croot, L.campaign_id(root_rel, 'seeds'), entries, L.configuration(),
                                             L.CAP, concurrency, L.AUTHORITY, L.REQUIRED_CONSECUTIVE_CYCLES, extra)
    with open(path) as handle:
        spec = json.load(handle)
    return croot, path, sha, spec


def run_seeds_split(root_abs, root_rel, lattice, croot, spec_path, spec_sha, spec, stub):
    """The seeds phase by the launcher's own resumable seeds launches until complete. -> list of outcomes."""
    outcomes = []
    for _ in range(MAX_LAUNCHES):
        ctx = H.CampaignContext(croot, spec_path, spec_sha, spec, log=lambda m: None)
        o = L.seeds_launch(croot=croot, root_abs=root_abs, root_rel=root_rel, spec=spec, spec_path=spec_path,
                           spec_sha256=spec_sha, lattice=lattice, harness_evaluate=stub, ctx=ctx, require_git=False,
                           log=lambda m: None, results_extra={'w183i_c_planted': True})
        outcomes.append(o)
        if o['status'] != 'suspended':
            break
    return outcomes


def plant_search_spec(root_abs, root_rel, lattice, arm, concurrency, sigma_q, g0):
    """The search phase spec as L.freeze writes it (domain entries; extra arm / cache_table / initial_incumbent),
    frozen by H.freeze_campaign_spec; the seed cache read by L.load_seed_cache from the planted seeds phase."""
    ok, cinfo, cache = L.load_seed_cache(root_abs, root_rel, arm, lattice, g0=g0 if arm == L.ALIAS_MODEL else None,
                                         require_git=False)
    if not ok:
        raise AssertionError(f'planted seed cache refused: {cinfo.get("reason")}')
    inc, inc_rec = L.initial_incumbent(lattice, cache, sigma_q)
    phase = f'search_{arm}'
    extra = {'campaign_script': L.SCRIPT_NAME, 'e1_root': root_rel, 'phase': phase, 'arm': arm,
             'cache_table': {k: L._jsonable(v) for k, v in sorted(cache.items())}, 'initial_incumbent': inc_rec,
             'w183i_c_planted': True}
    croot = L.phase_root(root_abs, root_rel, phase)
    path, sha, _spec = H.freeze_campaign_spec(croot, L.campaign_id(root_rel, phase), L.domain_entries(lattice, arm),
                                              L.configuration(), L.CAP, concurrency, L.AUTHORITY,
                                              L.REQUIRED_CONSECUTIVE_CYCLES, extra)
    with open(path) as handle:
        spec = json.load(handle)
    return croot, path, sha, spec, cache, inc


def reference_search(lattice, sigma_q, root_rel, croot, spec_path, spec_sha, spec, cache, inc, stub):
    """The IN-PROCESS reference: ONE call of L.run_continuation_e1 with the v2 --run evaluate_fn logic (run() ~2192:
    labels from keys, heartbeat tag batch_NNN per call, point_from_harness, cache_entry_from_point) over the stub."""
    arm = spec['extra']['arm']
    key_of = L.make_arm_key_of(lattice, arm)
    label_of_key = {e['eval_key']: e['label'] for e in spec['candidates']}
    by_label = {e['label']: e for e in spec['candidates']}
    ctx = H.CampaignContext(croot, spec_path, spec_sha, spec, log=lambda m: None)
    points, history, starts, logs, counter = {}, [], [], [], {'n': 0}

    def evaluate_fn(batch):
        keys = [key_of(z) for z in batch]
        labels = [label_of_key[k] for k in keys]
        counter['n'] += 1
        stub.last_batch_info = {}
        recs = stub(labels, ctx, heartbeat_tag=f'batch_{counter["n"]:03d}')
        out = []
        for k, lab, rec in zip(keys, labels, recs):
            p = L.point_from_harness(rec, by_label[lab], arm)
            points[lab] = p
            out.append(L.cache_entry_from_point(p, root_rel, spec['campaign_id']))
        return out

    n_calls0 = len(stub.calls)
    out = L.run_continuation_e1(lattice, copy.deepcopy(cache), key_of, dict(inc), evaluate_fn, sigma_q, k0=L.HALTON_K0,
                                max_new_evaluations=L.N_MAX_NEW_EVALUATIONS, max_polls=L.MAX_POLLS,
                                batch_size=int(spec['concurrency']), completion_cap=L.COMPLETION_CAP, log=logs.append,
                                on_poll=history.append, on_poll_start=lambda r: starts.append(r['poll_index']),
                                advance=True, advance_window=L.ADVANCE_WINDOW)
    return {'out': out, 'points': points, 'history': history, 'calls': copy.deepcopy(stub.calls[n_calls0:]),
            'logs': logs, 'on_poll_start': starts}


def split_search(lattice, sigma_q, root_abs, root_rel, croot, spec_path, spec_sha, spec, stub, unit, g0,
                 max_launches=MAX_LAUNCHES):
    """The split run: L.search_launch once per launch until it terminates (each call rebuilds from disk)."""
    outcomes = []
    n_calls0 = len(stub.calls)
    for _ in range(max_launches):
        o = launch_once(lattice, sigma_q, root_abs, root_rel, croot, spec_path, spec_sha, spec, stub, unit, g0)
        outcomes.append(o)
        if o['status'] != 'suspended':
            break
    return outcomes, copy.deepcopy(stub.calls[n_calls0:])


def launch_once(lattice, sigma_q, root_abs, root_rel, croot, spec_path, spec_sha, spec, stub, unit, g0):
    ctx = H.CampaignContext(croot, spec_path, spec_sha, spec, log=lambda m: None)
    return L.search_launch(croot=croot, root_abs=root_abs, root_rel=root_rel, spec=spec, spec_path=spec_path,
                           spec_sha256=spec_sha, lattice=lattice, sigma_q=sigma_q, harness_evaluate=stub, ctx=ctx,
                           launch_unit=unit, g0=g0, require_git=False, log=lambda m: None,
                           results_extra={'w183i_c_planted': True})


def tree_digest(base, sub=''):
    """{relative path: sha256} of every file under base/sub."""
    root = os.path.join(base, sub)
    return {os.path.relpath(f, base): H.sha256_file(f) for f in L.files_under(root)} if os.path.isdir(root) else {}


def read_launch_records(croot):
    out, n = [], 1
    while os.path.isfile(L.launch_paths(croot, n)[0]):
        with open(L.launch_paths(croot, n)[0]) as handle:
            out.append(json.load(handle))
        n += 1
    return out


def compare_split(ref, outcomes, calls, croot_ref_tree, croot_split, unit):
    """The split run against the in-process reference (bytes after the same writer)."""
    final = outcomes[-1]
    lrs = read_launch_records(croot_split)
    with open(os.path.join(croot_split, L.RESULTS_FILE)) as handle:
        results = json.load(handle)
    persisted_history = [p for lr in lrs for p in lr['completed_polls']]
    split_tree = tree_digest(croot_split, 'evals')
    ref_evals = {k: v for k, v in croot_ref_tree.items() if k.startswith('evals' + os.sep)}
    hb_ref = sorted(k for k in croot_ref_tree if k.startswith('campaign_heartbeat_'))
    hb_split = sorted(os.path.basename(f) for f in glob.glob(os.path.join(croot_split, 'campaign_heartbeat_*.json')))
    live_batches = [lr['batches'] for lr in lrs]
    res = {
        'n_launches': len(outcomes), 'statuses': [o['status'] for o in outcomes],
        'final_status_terminated': final['status'] == 'terminated',
        'return_dict_bytes_equal': _ser(ref['out']) == _ser(final.get('result')),
        'return_dict_bytes_equal_launch_record_on_disk': bool(lrs) and _ser(ref['out']) == _ser(lrs[-1]['result']),
        'poll_history_bytes_equal_concatenated_launch_records': (
            len(persisted_history) == len(ref['history'])
            and all(_ser(a) == _ser(b) for a, b in zip(ref['history'], persisted_history))),
        'poll_history_bytes_equal_results_file': _ser(ref['history']) == _ser(results['poll_history']),
        'points_bytes_equal_results_file': _ser(ref['points']) == _ser(results['points']),
        'stub_calls_equal': ref['calls'] == calls,
        'evals_tree_bytes_equal': ref_evals == split_tree and bool(split_tree) == bool(ref_evals),
        'heartbeat_names_equal': hb_ref == hb_split,
        'n_evaluations': len(ref['points']), 'n_batches': len(ref['calls']), 'n_polls': ref['out']['n_polls'],
        'termination': ref['out']['termination']['reason'],
        'live_batches_per_launch': [len(b) for b in live_batches],
        'ref_return_sha256': hashlib.sha256(_ser(ref['out']).encode()).hexdigest(),
        'split_return_sha256': hashlib.sha256(_ser(final.get('result')).encode()).hexdigest(),
    }
    if unit == 'batch':
        res['every_launch_one_live_batch'] = (all(n == 1 for n in res['live_batches_per_launch'])
                                              if res['n_batches'] else res['live_batches_per_launch'] == [0])
        res['n_launches_equals_n_batches'] = len(outcomes) == max(1, res['n_batches'])
    else:
        # every launch's live batches belong to one poll, and each evaluating poll is in exactly one launch
        poll_of_label = {}
        for i, h in enumerate(ref['history']):
            for b in h['batches']:
                for lab in b:
                    poll_of_label[lab] = i
        polls_per_launch = [sorted({poll_of_label[lattice_label_of(lab)] for b in lr['batches'] for lab in b})
                            for lr in lrs]
        evaluating = sorted({i for i, h in enumerate(ref['history']) if h['batches']})
        res['polls_per_launch'] = polls_per_launch
        res['every_launch_at_most_one_poll'] = all(len(p) <= 1 for p in polls_per_launch)
        res['every_evaluating_poll_in_exactly_one_launch'] = sorted(p[0] for p in polls_per_launch if p) == evaluating
    # the boundaries: where each launch stopped (from its state_after)
    res['boundaries'] = [{'launch': lr['launch_index'], 'terminated': lr['terminated'],
                          'state_k_next': lr['state_after'].get('k_next'),
                          'n_new': lr['state_after'].get('n_new'), 'n_barrier_new': lr['state_after'].get('n_barrier_new'),
                          'n_polls': lr['state_after'].get('n_polls'),
                          'poll_in_progress': lr['state_after'].get('poll_in_progress'),
                          'completed_polls': [(p['poll_index'], p['decision'],
                                               [s['k'] for s in p.get('halton_indices_skipped') or []])
                                              for p in lr['completed_polls']]} for lr in lrs]
    res['ok'] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


def boundary_after_advance(cmp_res):
    """(a) a launch ended inside an advanced poll (its poll_in_progress has skipped indices: k_next already past the
    advance) and the next launch continued it; (b) a launch's last completed poll was advanced and the next launch
    evaluated a further poll."""
    b = cmp_res['boundaries']
    inside = any((x['poll_in_progress'] or {}).get('skipped_ks') for x in b[:-1])
    after = any(x['completed_polls'] and x['completed_polls'][-1][2] and not x['terminated'] for x in b[:-1])
    return {'a_launch_ended_inside_an_advanced_poll': inside, 'a_launch_ended_right_after_an_advanced_poll': after}


# ======================================================================================================================
#  scenario runner
# ======================================================================================================================
class Scenario:
    def __init__(self, name, arm, concurrency, seed_q, search_q, barrier=(), units=('batch', 'poll')):
        self.name, self.arm, self.concurrency = name, arm, concurrency
        self.seed_q, self.search_q, self.barrier, self.units = seed_q, search_q, set(barrier), units


def build_base_root(scratch, sc, lattice, sigma_q, g0, g0_sha):
    """R = scratch/roots/<name> (the path every run of the scenario uses) and a pristine template copy."""
    root_rel = f'planted_{sc.name}'
    root_abs = os.path.join(scratch, 'roots', sc.name)
    os.makedirs(root_abs)
    plan = L.seed_plan(lattice)
    labels7 = [r['label'] for r in plan if r['disposition'] == 'evaluate']
    stub = StubHarness(lambda lab: sc.seed_q(lab), barrier=(), name=f'seeds_{sc.name}')
    croot, sp, ssha, spec = plant_seeds_spec(root_abs, root_rel, lattice, labels7, 4, g0_sha)
    outs = run_seeds_split(root_abs, root_rel, lattice, croot, sp, ssha, spec, stub)
    if outs[-1]['status'] != 'terminated':
        raise AssertionError(f'{sc.name}: planted seeds did not complete: {[o["status"] for o in outs]}')
    scroot, ssp, sssha, sspec, cache, inc = plant_search_spec(root_abs, root_rel, lattice, sc.arm, sc.concurrency,
                                                              sigma_q, g0)
    template = os.path.join(scratch, 'templates', sc.name)
    shutil.copytree(root_abs, template)
    return {'root_rel': root_rel, 'root_abs': root_abs, 'template': template, 'croot': scroot, 'spec_path': ssp,
            'spec_sha': sssha, 'spec': sspec, 'cache': cache, 'inc': inc,
            'seeds_launches': [o['status'] for o in outs]}


def reset_root(base, tag):
    """Move the current R aside to R__<tag> and restore R from the template."""
    if os.path.exists(base['root_abs']):
        os.rename(base['root_abs'], f"{base['root_abs']}__{tag}")
    shutil.copytree(base['template'], base['root_abs'])


def run_scenario(scratch, sc, lattice, sigma_q, g0, g0_sha):
    base = build_base_root(scratch, sc, lattice, sigma_q, g0, g0_sha)
    stub = StubHarness(sc.search_q, barrier=sc.barrier, name=f'search_{sc.name}')
    out = {'name': sc.name, 'arm': sc.arm, 'concurrency': sc.concurrency, 'initial_incumbent': base['inc']['label'],
           'seeds_launch_statuses': base['seeds_launches'], 'spec_sha256': base['spec_sha']}
    ref = reference_search(lattice, sigma_q, base['root_rel'], base['croot'], base['spec_path'], base['spec_sha'],
                           base['spec'], base['cache'], base['inc'], stub)
    ref_tree = tree_digest(base['croot'])
    reset_root(base, 'reference')
    runs = {'reference': {'out': ref['out'], 'points': ref['points'], 'calls': ref['calls'], 'logs': ref['logs']}}
    for unit in sc.units:
        outcomes, calls = split_search(lattice, sigma_q, base['root_abs'], base['root_rel'], base['croot'],
                                       base['spec_path'], base['spec_sha'], base['spec'], stub, unit, g0)
        cmp_res = compare_split(ref, outcomes, calls, ref_tree, base['croot'], unit)
        cmp_res.update(boundary_after_advance(cmp_res))
        # a further launch after the terminating one is refused (the phase is complete); nothing evaluated
        n0 = len(stub.calls)
        again = launch_once(lattice, sigma_q, base['root_abs'], base['root_rel'], base['croot'], base['spec_path'],
                            base['spec_sha'], base['spec'], stub, unit, g0)
        inv = L.root_inventory(base['root_abs'], base['root_rel'])
        cmp_res['launch_after_termination_refused'] = (again['status'] == 'refused' and len(stub.calls) == n0
                                                       and any('the phase is complete' in r
                                                               for r in again['refusals']))
        cmp_res['root_rule_resume_refuses_the_complete_phase'] = bool(
            L.root_refusals(inv, 'resume', f'search_{sc.arm}'))
        cmp_res['ok'] = cmp_res['ok'] and cmp_res['launch_after_termination_refused'] \
            and cmp_res['root_rule_resume_refuses_the_complete_phase']
        out[f'split_{unit}'] = cmp_res
        runs[f'split_{unit}'] = {'outcomes': [{k: v for k, v in o.items() if k != 'result'} for o in outcomes],
                                 'calls': calls, 'launch_records': read_launch_records(base['croot'])}
        _log(f"{sc.name} [{unit}]: {cmp_res['n_launches']} launches, {cmp_res['n_batches']} batches, "
             f"{cmp_res['n_polls']} polls, {cmp_res['termination']}; byte-equal ok={cmp_res['ok']}")
        reset_root(base, f'split_{unit}')
    return out, runs, base, stub


# ======================================================================================================================
#  item 2 / 3: planted foreign records and tampered state on a mid-run phase root
# ======================================================================================================================
def _rewrite_json(path, obj):
    with open(path, 'w') as handle:
        handle.write(GRIO.dumps(obj, indent=1, sort_keys=True))


def _rehash_manifest(croot, n):
    """Re-hash every file the manifest of launch n lists (so that only the targeted check can fire)."""
    rec_p, man_p = L.launch_paths(croot, n)
    with open(man_p) as handle:
        manifest = json.load(handle)
    manifest = {f: H.sha256_file(L._abs(f)) for f in manifest}
    _rewrite_json(man_p, manifest)


def _phase_snapshot(croot):
    return tree_digest(croot)


def attempt(case, apply_fn, expect_substrings, ctxs, rehash=True):
    """Restore the mid-run root, apply the tamper, attempt ONE launch: must be refused with every expected substring,
    0 stub calls, nothing written; then restore."""
    lattice, sigma_q, base, stub, unit, g0, backup = ctxs
    if os.path.exists(base['root_abs']):
        shutil.rmtree(base['root_abs'])
    shutil.copytree(backup, base['root_abs'])
    detail = apply_fn(base) or {}
    n_last = len(read_launch_records(base['croot']))
    if rehash and n_last:
        _rehash_manifest(base['croot'], n_last)
    before = _phase_snapshot(base['croot'])
    n0 = len(stub.calls)
    if base['phase'] == 'seeds':
        ctx = H.CampaignContext(base['croot'], base['spec_path'], base['spec_sha'], base['spec'], log=lambda m: None)
        o = L.seeds_launch(croot=base['croot'], root_abs=base['root_abs'], root_rel=base['root_rel'],
                           spec=base['spec'], spec_path=base['spec_path'], spec_sha256=base['spec_sha'],
                           lattice=lattice, harness_evaluate=stub, ctx=ctx, require_git=False, log=lambda m: None)
    else:
        o = launch_once(lattice, sigma_q, base['root_abs'], base['root_rel'], base['croot'], base['spec_path'],
                        base['spec_sha'], base['spec'], stub, unit, g0)
    after = _phase_snapshot(base['croot'])
    found = {s: any(s in r for r in o['refusals']) for s in expect_substrings}
    res = {'case': case, 'status': o['status'], 'refusals': o['refusals'], 'expected_substrings_found': found,
           'stub_calls_during_attempt': len(stub.calls) - n0, 'phase_root_unchanged': before == after,
           'detail': detail, 'manifest_rehashed': rehash}
    res['ok'] = (o['status'] == 'refused' and all(found.values()) and res['stub_calls_during_attempt'] == 0
                 and res['phase_root_unchanged'])
    _log(f"  {case}: {o['status']} ok={res['ok']} refusals={[r[:110] for r in o['refusals']][:3]}")
    shutil.rmtree(base['root_abs'])
    shutil.copytree(backup, base['root_abs'])
    return res


def _last_lr(base):
    n = len(read_launch_records(base['croot']))
    path = L.launch_paths(base['croot'], n)[0]
    with open(path) as handle:
        return n, path, json.load(handle)


def mid_run(scratch, base, stub, lattice, sigma_q, unit, g0, n_launches, tag):
    """Restore R from the template and run `n_launches` launches; back the mid-run root up (the tamper base)."""
    if os.path.exists(base['root_abs']):
        shutil.rmtree(base['root_abs'])
    shutil.copytree(base['template'], base['root_abs'])
    outs = []
    for _ in range(n_launches):
        outs.append(launch_once(lattice, sigma_q, base['root_abs'], base['root_rel'], base['croot'],
                                base['spec_path'], base['spec_sha'], base['spec'], stub, unit, g0))
        if outs[-1]['status'] != 'suspended':
            raise AssertionError(f'mid-run {tag}: launch {len(outs)} {outs[-1]["status"]} before {n_launches}')
    backup = os.path.join(scratch, 'backups', tag)
    shutil.copytree(base['root_abs'], backup)
    return backup, outs


def foreign_record_cases(ctxs, other):
    """Item 2. `other` = a completed root of the OTHER arm ({'croot', 'spec', 'records'})."""
    lattice, sigma_q, base, stub, unit, g0, backup = ctxs
    out = []

    def wrong_root(b):
        n, path, lr = _last_lr(b)
        lab = lr['batches'][-1][0]
        ref_root = f"{b['root_abs']}__reference"
        moved = lr['new_points'][lab]['eval_dir'].replace(_rel(b['root_abs']), _rel(ref_root))
        if not os.path.isdir(L._abs(moved)):
            raise AssertionError(f'the sibling reference eval dir does not exist: {moved}')
        lr['new_points'][lab]['eval_dir'] = moved
        _rewrite_json(path, lr)
        return {'label': lab, 'eval_dir_now': moved, 'sibling_has_identical_bytes': True}
    out.append(attempt('wrong_root: a point whose eval dir lies in a sibling root (identical bytes there)', wrong_root,
                       ['eval_dir_inside_the_phase_root', 'eval_dir_is_the_spec_entrys'], ctxs))

    def wrong_key(b):
        n, path, lr = _last_lr(b)
        lab = lr['batches'][-1][0]
        p = lr['new_points'][lab]
        z = lattice.z_of_canonical(p['candidate_canonical'])
        other_model = 'N' if b['arm'] == 'A' else 'A'
        p['eval_key'] = L.make_arm_key_of(lattice, other_model)(z)
        _rewrite_json(path, lr)
        return {'label': lab, 'eval_key_now': p['eval_key'], 'which': f'its key under model {other_model}'}
    out.append(attempt('wrong_key: a point whose eval key is not its key under the arm\'s model', wrong_key,
                       ['eval_key_recomputes_under_the_arms_model', 'eval_key_equals_the_spec_entrys'], ctxs))

    def hash_mismatch(b):
        n, path, lr = _last_lr(b)
        lab = lr['batches'][-1][0]
        rec_path = os.path.join(L._abs(lr['new_points'][lab]['eval_dir']), 'evaluation_record.json')
        with open(rec_path) as handle:
            rec = json.load(handle)
        rec['certified_cost'] = (rec['certified_cost'] or 0.0) - 1.0e6
        _rewrite_json(rec_path, rec)
        return {'label': lab, 'record': _rel(rec_path), 'certified_cost_now': rec['certified_cost']}
    out.append(attempt('hash_mismatch: an evaluation record whose bytes no longer match its launch manifest',
                       hash_mismatch, ['every_manifest_hash_holds', 'record_file_hashed_in_this_launchs_manifest'],
                       ctxs, rehash=False))

    def _plant_other(b, relabel):
        n, path, lr = _last_lr(b)
        olr = next(lr for lr in reversed(other['records']) if lr['batches'])
        olab = olr['batches'][-1][0]
        op = copy.deepcopy(olr['new_points'][olab])
        src_dir = L._abs(op['eval_dir'].replace(_rel(other['root_abs']), _rel(other['renamed_root_abs'])))
        dst_dir = os.path.join(b['croot'], 'evals', os.path.basename(src_dir))
        shutil.copytree(src_dir, dst_dir)
        op['eval_dir'] = _rel(dst_dir)
        lab = olab
        if relabel:
            lab = lattice_label_of(olab) + f"_{b['arm']}_{L.FLEX_M:g}x"
        lr['new_points'][lab] = op
        lr['batches'].append([lab])
        _rewrite_json(path, lr)
        man_p = L.launch_paths(b['croot'], n)[1]
        with open(man_p) as handle:
            manifest = json.load(handle)
        for f in L.files_under(dst_dir):
            manifest[_rel(f)] = H.sha256_file(f)
        _rewrite_json(man_p, manifest)
        return {'other_arm_label': olab, 'planted_as': lab, 'other_arm_model': op.get('model'),
                'planted_eval_dir': op['eval_dir']}
    out.append(attempt('other_arm: a record of the other arm (its own label), files and manifest made consistent',
                       lambda b: _plant_other(b, False),
                       ['label_in_the_spec', 'point_model_is_the_arms_model'], ctxs))
    out.append(attempt('other_arm_relabelled: a record of the other arm under this arm\'s label of the same candidate',
                       lambda b: _plant_other(b, True),
                       ['point_model_is_the_arms_model', 'eval_key_recomputes_under_the_arms_model',
                        'eval_dir_is_the_spec_entrys'], ctxs))

    def unlisted(b):
        ref_root = f"{b['root_abs']}__reference"
        ref_croot = b['croot'].replace(b['root_abs'], ref_root)
        done = {lab for lr in read_launch_records(b['croot']) for bb in lr['batches'] for lab in bb}
        by_label = {e['label']: e for e in b['spec']['candidates']}
        cand = sorted(d for d in os.listdir(os.path.join(ref_croot, 'evals'))
                      if not any(by_label[lab]['eval_dir'] == d for lab in done))
        shutil.copytree(os.path.join(ref_croot, 'evals', cand[0]), os.path.join(b['croot'], 'evals', cand[0]))
        return {'planted_eval_dir': cand[0]}
    out.append(attempt('unlisted: an eval dir (copied from the reference run) that no launch manifest lists', unlisted,
                       ['foreign'], ctxs))

    def foreign_launch_record(b):
        n, path, lr = _last_lr(b)
        lr['e1_root'] = 'planted_some_other_root'
        _rewrite_json(path, lr)
        return {'e1_root_now': lr['e1_root']}
    out.append(attempt('foreign_launch_record: the last launch record names another E1 root', foreign_launch_record,
                       ['e1_root_is_this_root'], ctxs))
    return out


def state_tamper_cases(ctxs):
    """Item 3: the last launch record's state (manifest re-hashed, so only the disagreement check can fire)."""
    lattice, sigma_q, base, stub, unit, g0, backup = ctxs
    out = []

    def _tamper(field, fn):
        def apply(b):
            n, path, lr = _last_lr(b)
            before = copy.deepcopy(lr['state_after'].get(field))
            fn(lr['state_after'])
            _rewrite_json(path, lr)
            return {'field': field, 'before': before, 'after': lr['state_after'].get(field)}
        return apply
    out.append(attempt('k_next +1', _tamper('k_next', lambda s: s.__setitem__('k_next', s['k_next'] + 1)),
                       ['k_next_equals_the_last_launch_record'], ctxs))
    out.append(attempt('budget: n_new +1', _tamper('n_new', lambda s: s.__setitem__('n_new', s['n_new'] + 1)),
                       ['n_new_equals_the_last_launch_record'], ctxs))
    out.append(attempt('barrier count: n_barrier_new +1',
                       _tamper('n_barrier_new', lambda s: s.__setitem__('n_barrier_new', s['n_barrier_new'] + 1)),
                       ['n_barrier_new_equals_the_last_launch_record'], ctxs))
    out.append(attempt('barrier count: n_barrier_new -1 (floored at 0 -> +1 if 0)',
                       _tamper('n_barrier_new', lambda s: s.__setitem__(
                           'n_barrier_new', s['n_barrier_new'] - 1 if s['n_barrier_new'] > 0 else 1)),
                       ['n_barrier_new_equals_the_last_launch_record'], ctxs))
    out.append(attempt('incumbent label', _tamper('incumbent', lambda s: s['incumbent'].__setitem__(
        'label', s['incumbent']['label'] + '_tampered')), ['incumbent_equals_the_last_launch_record'], ctxs))
    out.append(attempt('incumbent F -1 EUR', _tamper('incumbent', lambda s: s['incumbent'].__setitem__(
        'F', s['incumbent']['F'] - 1.0)), ['incumbent_equals_the_last_launch_record'], ctxs))
    out.append(attempt('poll in progress: batches_done dropped', _tamper('poll_in_progress', lambda s: (
        s['poll_in_progress'].__setitem__('batches_done', s['poll_in_progress']['batches_done'][:-1] or [['x']])
        if s.get('poll_in_progress') else s.__setitem__('poll_in_progress', {'tampered': True}))),
        ['poll_in_progress_equals_the_last_launch_record'], ctxs))

    def poll_record(b):
        for lr_i in range(len(read_launch_records(b['croot'])), 0, -1):
            path = L.launch_paths(b['croot'], lr_i)[0]
            with open(path) as handle:
                lr = json.load(handle)
            if lr['completed_polls']:
                lr['completed_polls'][-1]['n_cache_hits'] += 1
                _rewrite_json(path, lr)
                _rehash_manifest(b['croot'], lr_i)
                if lr_i < len(read_launch_records(b['croot'])):   # the chain: re-link the next record too
                    for j in range(lr_i + 1, len(read_launch_records(b['croot'])) + 1):
                        pj = L.launch_paths(b['croot'], j)[0]
                        with open(pj) as handle:
                            lj = json.load(handle)
                        lj['previous_launch_manifest_sha256'] = H.sha256_file(L.launch_paths(b['croot'], j - 1)[1])
                        _rewrite_json(pj, lj)
                        _rehash_manifest(b['croot'], j)
                return {'launch': lr_i, 'poll_index': lr['completed_polls'][-1]['poll_index'],
                        'field': 'n_cache_hits +1', 'chain_relinked': True}
        raise AssertionError('no completed poll to tamper')
    out.append(attempt('a persisted poll record (n_cache_hits +1; manifests re-hashed, chain re-linked)', poll_record,
                       ['replayed_polls_byte_equal_the_persisted_polls'], ctxs))
    out.append(attempt('k_next +1 WITHOUT re-hashing the manifest',
                       _tamper('k_next', lambda s: s.__setitem__('k_next', s['k_next'] + 1)),
                       ['manifest_lists_the_launch_record_and_its_hash_holds'], ctxs, rehash=False))
    return out


# ======================================================================================================================
#  item 4: the source diff (W183g (b) method + the resume-hook category)
# ======================================================================================================================
def source_diff():
    old = inspect.getsource(C.run_continuation).splitlines()
    new = inspect.getsource(L.run_continuation_e1).splitlines()
    unified = '\n'.join(difflib.unified_diff(old, new, 'p515_s53_f2_certificate.run_continuation',
                                             'p515_s55_e1_launcher.run_continuation_e1', lineterm='', n=2))
    header_old = next(i for i, ln in enumerate(old) if ln.rstrip().endswith('):')) + 1
    header_new = next(i for i, ln in enumerate(new) if ln.rstrip().endswith('):')) + 1
    b0 = next(i for i, ln in enumerate(new) if ln.strip().startswith('# >>> B+ advance'))
    b1 = next(i for i, ln in enumerate(new) if ln.strip().startswith('# <<< B+ advance'))
    r0 = next(i for i, ln in enumerate(new) if '# >>> resume hook' in ln)
    r1 = next(i for i, ln in enumerate(new) if '# <<< resume hook' in ln)
    k_state_added = {'    k_next = k0', '        k = k_next'}
    k_state_removed = {'        k = k0 + i'}
    rows, bad = [], []
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        for i in range(i1, i2):
            cat = ('signature' if i < header_old else 'k_next_state' if old[i] in k_state_removed else None)
            rows.append({'side': '-', 'line_no': i + 1, 'text': old[i], 'category': cat})
            if cat is None:
                bad.append(rows[-1])
        for j in range(j1, j2):
            if j < header_new:
                cat = 'signature'
            elif new[j] in k_state_added:
                cat = 'k_next_state'
            elif r0 <= j <= r1:
                cat = 'resume_hook'
            elif b0 <= j <= b1:
                cat = ('inserted_block:k_next_state' if new[j].strip() == 'k_next = k + 1' else 'inserted_block')
            else:
                cat = None
            rows.append({'side': '+', 'line_no': j + 1, 'text': new[j], 'category': cat})
            if cat is None:
                bad.append(rows[-1])
    counts = {}
    for r in rows:
        counts[f"{r['side']}{r['category']}"] = counts.get(f"{r['side']}{r['category']}", 0) + 1
    return {'unified_diff': unified, 'classified_lines': rows, 'counts_by_side_and_category': counts,
            'uncategorised_lines': bad, 'only_declared_categories': not bad,
            'resume_hook_lines': [r['text'] for r in rows if r['category'] == 'resume_hook'],
            'signature_lines_added': [r['text'] for r in rows if r['category'] == 'signature' and r['side'] == '+'],
            'body_after_block_identical': old[old.index('        cands = record[\'candidates\']'):]
            == new[new.index('        cands = record[\'candidates\']'):],
            'docstring_identical': old[header_old:header_old + 3] == new[header_new:header_new + 3],
            'old_sha256': hashlib.sha256('\n'.join(old).encode()).hexdigest(),
            'new_sha256': hashlib.sha256('\n'.join(new).encode()).hexdigest()}


def w183g_b_checks_on_v3(lattice, sigma_q, table, w183g_b_out):
    """Items 1-3 of the W183g (b) gate, re-run on v3 with its own functions (G.paired / compare_*), plus 'a probe
    changes nothing'. The window-exhaustion incumbent is W183g (b)'s committed first hit (the scan is a property of the
    unchanged C.build_poll; not re-scanned)."""
    seeds = {s: z for s, z, _n, _y, _t in L.SEEDS}
    checks, detail, runs_out = {}, {}, {}
    for arm in L.MODELS:
        key_of = L.make_arm_key_of(lattice, arm)
        for seed, z in seeds.items():
            case = f'{seed}_{arm}'
            stub = G.LabelStub(lattice, key_of, G.q_hash, barrier=(), name=f'q_hash_{case}')
            inc, cache = G.incumbent_and_cache(lattice, key_of, stub, z, seeds.values())
            kw = {'k0': L.HALTON_K0, 'max_new_evaluations': L.N_MAX_NEW_EVALUATIONS, 'max_polls': L.MAX_POLLS,
                  'batch_size': 1, 'completion_cap': L.COMPLETION_CAP}
            rec, runs = G.paired(f'seed case {case}', lattice, key_of, stub, inc, cache, sigma_q, kw)
            runs_out[f'item4_seed_{case}'] = runs
            on0 = G.first_record(runs['on'])
            imp0 = G.first_record(runs['imported'])
            polled = [c for c in on0['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]
            exp = G.EXPECTED_B_PLUS[seed]
            if seed == 'x0':
                exp_poll0 = (imp0['completion_triggered'] is True and imp0['completion']['n_feasible'] == 14
                             and not imp0['completion']['over_cap'])
            else:
                exp_poll0 = (imp0['decision'] == 'stopped_for_review_completion_cap'
                             and imp0['completion']['n_feasible'] == G.EXPECTED_STORAGE_REFUSAL_COMPLETION[seed]
                             and runs['imported']['calls'] == [])
            by_k = {r['k']: r for r in table[case]['same_incumbent_at_k_0_to_9']}
            skipped = on0['halton_indices_skipped']
            for k, v in rec['off_vs_imported'].items():
                if isinstance(v, bool):
                    checks[f'seed_{case}:off:{k}'] = v
            checks[f'seed_{case}:expected_imported_poll0'] = exp_poll0
            checks[f'seed_{case}:on_k'] = on0['poll_index'] == exp['k']
            checks[f'seed_{case}:on_size'] = len(polled) == exp['size']
            checks[f'seed_{case}:on_skipped'] = [s['k'] for s in skipped] == exp['skipped']
            checks[f'seed_{case}:on_completion_use'] = on0['completion_triggered'] is exp['completion_used']
            checks[f'seed_{case}:on_n_distinct_equals_table'] = (on0['n_distinct_feasible_poll_points']
                                                                 == by_k[on0['poll_index']]['n_distinct'])
            checks[f'seed_{case}:on_vs_imported_ok'] = rec['on_vs_imported']['ok']
            # a probe passed changes nothing (bytes)
            probe = {}
            with_probe = G._run(L.run_continuation_e1, stub, lattice, cache, key_of, inc, sigma_q, kw,
                                {'advance': True, 'advance_window': L.ADVANCE_WINDOW, 'state_probe': probe})
            checks[f'seed_{case}:probe_changes_nothing_bytes'] = (
                _ser(with_probe['out']) == _ser(runs['on']['out']) and with_probe['calls'] == runs['on']['calls']
                and with_probe['logs'] == runs['on']['logs'])
            st = probe['get']()
            checks[f'seed_{case}:probe_state_equals_return'] = (
                st['n_new'] == with_probe['out']['n_new_evaluations'] and st['n_polls'] == with_probe['out']['n_polls']
                and st['n_barrier_new'] == with_probe['out']['n_barrier_new_evaluations']
                and st['incumbent']['eval_key'] == with_probe['out']['incumbent']['eval_key'])
            detail[case] = {'on_first_poll_k': on0['poll_index'], 'size': len(polled),
                            'skipped': [s['k'] for s in skipped], 'expected': exp,
                            'on_polls_k': [h['poll_index'] for h in runs['on']['out']['history']],
                            'on_termination': runs['on']['out']['termination']['reason']}
            _log(f"  W183g(b) seed case {case} on v3: k {on0['poll_index']} size {len(polled)} skipped "
                 f"{[s['k'] for s in skipped]}")
    reproduced = L.first_poll_by_seed(lattice, sigma_q)
    checks['k_0_9_table_reproduced'] = json.loads(_ser(reproduced)) == table
    # the item-3 paths of W183g (b) (code as in G.main, ~576-715), on v3
    key_a = L.make_arm_key_of(lattice, 'A')
    z0, z_unit = seeds['x0'], seeds['unit']
    std = {'k0': L.HALTON_K0, 'max_new_evaluations': L.N_MAX_NEW_EVALUATIONS, 'max_polls': L.MAX_POLLS,
           'completion_cap': L.COMPLETION_CAP}
    paths = {}

    def _scenario(name, stub, z_inc, kw):
        inc, cache = G.incumbent_and_cache(lattice, key_a, stub, z_inc, [z_inc])
        rec, runs = G.paired(name, lattice, key_a, stub, inc, cache, sigma_q, kw)
        runs_out[f'item4_path_{name}'] = runs
        paths[name] = rec
        return rec, runs
    all_labels = {lattice.label(z) for z in lattice.domain()}
    rec, runs = _scenario('A7_per_poll', G.LabelStub(lattice, key_a, lambda lab: Q0, barrier=all_labels - {'x0'},
                                                     name='all_barrier'), z0, dict(std, batch_size=1))
    rec['target'] = {'stop_barrier': runs['on']['out']['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule',
                     'barrier_this_poll_2': runs['on']['out']['termination'].get('barrier_new_this_poll') == 2}
    pts0 = G.poll_points(lattice, key_a, z0, 0, sigma_q)[1]
    p1 = next(z for z in pts0 if lattice.label(z) == 'y2030__n7_p0.25_e0.5__n9_p0.25_e0.5')
    pts1 = G.poll_points(lattice, key_a, p1, 1, sigma_q)[1]
    p2 = next(z for z in pts1 if z not in pts0 and z != z0 and (G.poll_points(lattice, key_a, z, 2, sigma_q)[1] or [])
              and G.poll_points(lattice, key_a, z, 2, sigma_q)[0]['completion_triggered'] is False)
    pts2 = G.poll_points(lattice, key_a, p2, 2, sigma_q)[1]
    b0 = next(z for z in pts0 if z not in (p1,))
    b1 = next(z for z in pts1 if z not in pts0 and z not in (p2, z0))
    b2 = next(z for z in pts2 if z not in pts0 and z not in pts1 and z not in (z0, p1))
    q_b = {lattice.label(p1): Q0 - 3.0e6, lattice.label(p2): Q0 - 6.0e6}
    rec, runs = _scenario('A7_overall', G.LabelStub(lattice, key_a, lambda lab: q_b.get(lab, Q0 + 1.0e6),
                                                    barrier={lattice.label(b) for b in (b0, b1, b2)},
                                                    name='one_barrier_per_poll'), z0, dict(std, batch_size=1))
    rec['target'] = {'stop_barrier': runs['on']['out']['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule',
                     'overall_3': runs['on']['out']['termination'].get('barrier_new_overall') == 3,
                     'three_polls': runs['on']['out']['n_polls'] == 3}
    p_st = next(z for z in pts0 if lattice.label(z) == 'y2030__n9_p0.25_e0.5')
    pts_k2 = G.poll_points(lattice, key_a, p_st, 2, sigma_q)[1]
    p_next = next(z for z in pts_k2 if z != z0 and z not in pts0 and lattice.has_storage(z))
    q_c = {lattice.label(p_st): Q0 - 3.0e6, lattice.label(p_next): Q0 - 6.0e6}
    rec, runs = _scenario('accepted_point_moves_incumbent_to_storage',
                          G.LabelStub(lattice, key_a, lambda lab: q_c.get(lab, Q0 + 1.0e6), name='move_to_storage'),
                          z0, dict(std, batch_size=4))
    hon = runs['on']['out']['history']
    rec['target'] = {'imported_cap_stop_at_k1': runs['imported']['out']['termination']['reason']
                     == 'STOP_FOR_REVIEW_completion_cap',
                     'on_poll_1_examines_k1_and_advances': len(hon) > 1 and hon[1]['advance_rule']['first_index'] == 1
                     and hon[1]['advance_rule']['n_examined'] > 1,
                     'on_next_from_last_polled_plus_1': len(hon) > 2
                     and hon[2]['advance_rule']['first_index'] == hon[1]['poll_index'] + 1}
    rec, runs = _scenario('budget_zero_x0', G.LabelStub(lattice, key_a, G.q_hash, name='q_hash'), z0,
                          dict(std, batch_size=1, max_new_evaluations=0))
    rec['target'] = {'budget_refusal_all': all(runs[x]['out']['termination']['reason'] == 'evaluation_budget_exhausted'
                                               for x in ('imported', 'off', 'on'))}
    rec, runs = _scenario('budget_5_unit', G.LabelStub(lattice, key_a, G.q_hash, name='q_hash'), z_unit,
                          dict(std, batch_size=1, max_new_evaluations=5))
    h0 = runs['on']['out']['history'][0]
    rec['target'] = {'on_polled_k2_budget_refusal': h0['poll_index'] == 2
                     and h0['decision'] == 'not_launched_evaluation_budget',
                     'on_skipped_0_1': [s['k'] for s in h0['halton_indices_skipped']] == [0, 1],
                     'on_nothing_evaluated': runs['on']['calls'] == []}
    q_e = {lattice.label(z): Q0 - 5.0e6 for z in pts0[:1]}
    rec, runs = _scenario('max_polls_1_x0', G.LabelStub(lattice, key_a, lambda lab: q_e.get(lab, Q0 + 1.0e6),
                                                        name='improving_x0'), z0, dict(std, batch_size=4, max_polls=1))
    rec['target'] = {'success_then_max_polls': all(runs[x]['out']['termination']['reason'] == 'max_polls_reached'
                                                   for x in ('imported', 'off', 'on'))}
    unit_k2 = G.poll_points(lattice, key_a, z_unit, 2, sigma_q)[1]
    q_e2 = {lattice.label(z): Q0 - 5.0e6 for z in unit_k2[:1]}
    rec, runs = _scenario('max_polls_1_unit', G.LabelStub(lattice, key_a, lambda lab: q_e2.get(lab, Q0 + 1.0e6),
                                                          name='improving_unit'), z_unit,
                          dict(std, batch_size=4, max_polls=1))
    rec['target'] = {'on_success_k2_then_max_polls': runs['on']['out']['termination']['reason'] == 'max_polls_reached'
                     and runs['on']['out']['history'][0]['poll_index'] == 2}
    hits = w183g_b_out['item3_window_exhaustion_scan']['hits']
    hit = min(hits, key=lambda h: (h['window_first_k'], h['label']))
    rec, runs = _scenario('window_exhaustion_iii_production_window_10',
                          G.LabelStub(lattice, key_a, G.q_hash, name='q_hash'), tuple(hit['z']),
                          dict(std, batch_size=1, k0=hit['window_first_k']))
    h0 = runs['on']['out']['history'][0]
    rec['target'] = {'all_three_cap_stop': all(runs[x]['out']['termination']['reason']
                                               == 'STOP_FOR_REVIEW_completion_cap' for x in ('imported', 'off', 'on')),
                     'n_examined_10_polled_none': h0['advance_rule']['n_examined'] == 10
                     and h0['advance_rule']['polled_index'] is None,
                     'skipped_the_9_further': [s['k'] for s in h0['halton_indices_skipped']]
                     == list(range(hit['window_first_k'] + 1, hit['window_first_k'] + 10))}
    rec['design'] = {'incumbent': hit, 'source': 'W183g (b) committed scan, first hit by window start then label'}
    for name, r in paths.items():
        checks[f'path_{name}:off_bytes_equal'] = r['off_vs_imported'][
            'return_dict_bytes_equal_after_removing_the_two_fields']
        for k in ('two_fields_trivial_on_every_record', 'stub_calls_equal', 'log_lines_equal', 'on_poll_start_equal',
                  'on_poll_equal'):
            checks[f'path_{name}:off_{k}'] = r['off_vs_imported'][k]
        checks[f'path_{name}:on_vs_imported_ok'] = r['on_vs_imported']['ok']
        for k, v in r['target'].items():
            checks[f'path_{name}:target:{k}'] = v
    detail['paths'] = {n: {'off_vs_imported': r['off_vs_imported'], 'on_vs_imported': r['on_vs_imported'],
                           'target': r['target'], 'summary': r['summary']} for n, r in paths.items()}
    diff = source_diff()
    checks['source_diff_only_declared_categories'] = diff['only_declared_categories']
    checks['source_diff_body_after_block_identical'] = diff['body_after_block_identical']
    checks['source_diff_docstring_identical'] = diff['docstring_identical']
    checks['source_diff_resume_hook_is_3_lines'] = len(diff['resume_hook_lines']) == 3
    detail['source_diff'] = diff
    return checks, detail, runs_out


# ======================================================================================================================
#  the per-launch time bound (JSON reads of the on-disk evaluation records; no solve)
# ======================================================================================================================
def time_bound():
    recs = sorted(glob.glob(os.path.join('data', 'SRP1', 'Results', 'P51*', '**', 'evaluation_record.json'),
                            recursive=True))
    rows = []
    for f in recs:
        try:
            with open(f) as handle:
                d = json.load(handle)
        except (OSError, ValueError):
            continue
        w = d.get('wall_time_s')
        w = w.get('child_process_s') if isinstance(w, dict) else w
        if not isinstance(w, (int, float)) or d.get('cap') != L.CAP:
            continue
        if any(d.get(k) for k in L.WRAPPER_KEYS):
            continue
        rows.append({'path': f, 'sha256': H.sha256_file(f), 'wall_s': w, 'cycles_run': d.get('cycles_run'),
                     'status': d.get('status'), 'flex_price_multiplier': d.get('flex_price_multiplier'),
                     'tail': bool((d.get('configuration') or {}).get('convergence_depth_tail'))})
    at2 = [r for r in rows if r['flex_price_multiplier'] == L.FLEX_M]

    def _stats(sel):
        if not sel:
            return None
        ws = sorted(r['wall_s'] for r in sel)
        spc = sorted(r['wall_s'] / r['cycles_run'] for r in sel if r['cycles_run'])
        return {'n': len(ws), 'max_h': ws[-1] / 3600, 'median_h': statistics.median(ws) / 3600,
                'p90_h': ws[int(0.9 * (len(ws) - 1))] / 3600, 'max_cycles': max(r['cycles_run'] or 0 for r in sel),
                's_per_cycle_min': spc[0] if spc else None, 's_per_cycle_max': spc[-1] if spc else None}
    s2 = _stats(at2)
    return {'source': ('on-disk evaluation_record.json under data/SRP1/Results/P51*/ (untracked eval dirs; each '
                       'file hashed below), cap 500, no settling / continuation wrapper (production exit)'),
            'n_files_scanned': len(recs), 'all_cap500_production_exit': _stats(rows), 'at_flex_2x': s2,
            'cap_500_worst_case_h_at_the_2x_max_s_per_cycle': (L.CAP * s2['s_per_cycle_max'] / 3600) if s2 else None,
            'rows_2x': at2, 'rows_all_paths_sha256': {r['path']: r['sha256'] for r in rows}}


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description='W183i (c) zero-solve gate of the E1 launcher v3 (resumable launches)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--g0-artifact', required=True)
    ap.add_argument('--g0-sha256', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--allow-uncommitted', action='store_true', help='smoke only: provenance pins not asserted')
    args = ap.parse_args()
    started = time.time()
    alive = own_copies_alive()
    if alive:
        raise SystemExit(f'REFUSED: another copy of this gate, the W183g (b) gate or the launcher is alive: {alive}')
    out_rel, out_abs = L.resolve_root(args.out)
    names = sorted(os.listdir(out_abs)) if os.path.isdir(out_abs) else None
    if names is None or [n for n in names if n != 'launch.log']:
        raise SystemExit(f'REFUSED: --out must be a fresh directory holding only launch.log: {names}')
    if (args.g0_artifact, args.g0_sha256) != (G.G0_FINAL['path'], G.G0_FINAL['sha256']):
        raise SystemExit(f'REFUSED: the G0 pointer must be the final-head one: {G.G0_FINAL}')
    scratch = os.path.realpath(os.path.abspath(args.scratch))
    if scratch.startswith(REPO + os.sep) or os.path.exists(scratch):
        raise SystemExit(f'REFUSED: --scratch must be a NEW directory outside the repository: {scratch}')
    os.makedirs(scratch)
    gate_scratch = os.path.join(scratch, 'gate')
    os.makedirs(gate_scratch)
    res = {'schema': 'p515_s55_w183i_c_e1_v3_resume_gate_v1', 'started_utc': L._utc(), 'argv': sys.argv,
           'interpreter': sys.executable, 'git_head': H._git(['rev-parse', 'HEAD']),
           'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 3 (iii)-(v); W183i order item (c)'],
           'objective_convention': ('no objective value of the problem is computed: every Q is a synthetic stub value '
                                    '(F = I + Q as the loop defines it, Q = gross_operational_cost convention)'),
           'zero_solves': 'SolveProfileGuard(permitted=()) before any other project import; verified at 0',
           'resume_rule': L.RESUME_RULE}
    checks = {}
    # ---- 0. provenance ----
    prov = {'launcher_sha256': _sha(L.SCRIPT_NAME), 'launcher_v2_sha256': LAUNCHER_V2_SHA256,
            'gate_sha256': _sha(SCRIPT_NAME), 'w183g_b_gate': dict(W183G_B_GATE, sha256_on_disk=_sha(
                W183G_B_GATE['path'])), 'w183g_b_out': dict(W183G_B_OUT, sha256_on_disk=_sha(W183G_B_OUT['path'])),
            'w183b_dryrun': dict(G.W183B_DRYRUN, sha256_on_disk=_sha(G.W183B_DRYRUN['path'])),
            'harness_sha256': _sha('p515_s44_campaign_harness.py'), 'harness_expected': HARNESS_SHA256,
            'unchanged_modules': {m: {'sha256': _sha(m), 'git_clean': not H._git(['status', '--porcelain', '--', m]),
                                      'tracked': bool(H._git(['ls-files', '--', m]))} for m in UNCHANGED_MODULES},
            'launcher_and_gate_committed_clean': not H._git(['status', '--porcelain', '--', L.SCRIPT_NAME,
                                                             SCRIPT_NAME])
            and bool(H._git(['ls-files', '--', L.SCRIPT_NAME])) and bool(H._git(['ls-files', '--', SCRIPT_NAME]))}
    res['provenance'] = prov
    checks['provenance'] = {
        'launcher_is_v3_not_v2': prov['launcher_sha256'] != LAUNCHER_V2_SHA256,
        'w183g_b_gate_sha256_pinned': prov['w183g_b_gate']['sha256_on_disk'] == W183G_B_GATE['sha256'],
        'w183g_b_out_sha256_pinned': prov['w183g_b_out']['sha256_on_disk'] == W183G_B_OUT['sha256'],
        'w183b_dryrun_sha256_pinned': prov['w183b_dryrun']['sha256_on_disk'] == G.W183B_DRYRUN['sha256'],
        'harness_is_the_patched_d9a944f2': prov['harness_sha256'] == HARNESS_SHA256,
        'certificate_pb_harness_hooks_unchanged_clean': all(v['git_clean'] and v['tracked']
                                                            for v in prov['unchanged_modules'].values())}
    if not args.allow_uncommitted:
        checks['provenance']['launcher_and_gate_committed_clean'] = prov['launcher_and_gate_committed_clean']
    with open(G.W183B_DRYRUN['path']) as handle:
        table = json.load(handle)['first_poll_if_each_seed_were_the_incumbent']
    with open(W183G_B_OUT['path']) as handle:
        w183g_b_out = json.load(handle)
    lattice, lev, lfail = L.build_lattice()
    if lfail:
        raise SystemExit(f'REFUSED (search tables): {lfail}')
    sigma_q = lev['sigma_Q']['sigma_Q_eur']
    g0 = (True, {'sha256_on_disk': args.g0_sha256})
    plan = L.seed_plan(lattice)
    labels7 = [r['label'] for r in plan if r['disposition'] == 'evaluate']
    runs_out = {}

    # ---- 1a. the seeds phase split in batches of its own size, against one unsplit call ----
    _log('item 1a: the seeds phase split in batches of its own size ...')
    seed_q = {lab: G.q_hash(lab) for lab in labels7}
    item1_seeds = {}
    for name, labels, conc in (('seven_at_1', labels7, 1), ('seven_at_4', labels7, 4),
                               ('batch_root_four_at_4', list(L.BATCH_SEEDS), 4)):
        root_rel = f'planted_seeds_{name}'
        root_abs = os.path.join(gate_scratch, 'seeds', name)
        os.makedirs(root_abs)
        stub = StubHarness(lambda lab: seed_q[lab], name=f'seeds_{name}')
        croot, sp, ssha, spec = plant_seeds_spec(root_abs, root_rel, lattice, labels, conc, args.g0_sha256)
        template = os.path.join(gate_scratch, 'seeds_templates', name)
        shutil.copytree(root_abs, template)
        # reference: ONE call over every label (v2 run(): H.evaluate(labels, ctx, heartbeat_tag='seeds'))
        ctx = H.CampaignContext(croot, sp, ssha, spec, log=lambda m: None)
        seed_of = {r['label']: (r['seed'], r['model']) for r in plan}
        by_label = {e['label']: e for e in spec['candidates']}
        all_labels = [e['label'] for e in spec['candidates']]
        recs = stub(all_labels, ctx, heartbeat_tag='seeds')
        ref_points = {lab: L.point_from_harness(rec, by_label[lab], seed_of[lab][1], seed_of[lab][0])
                      for lab, rec in zip(all_labels, recs)}
        ref_tree = tree_digest(croot, 'evals')
        ref_calls = copy.deepcopy(stub.calls)
        os.rename(root_abs, root_abs + '__reference')
        shutil.copytree(template, root_abs)
        outs = run_seeds_split(root_abs, root_rel, lattice, croot, sp, ssha, spec, stub)
        split_calls = stub.calls[len(ref_calls):]
        with open(os.path.join(croot, L.RESULTS_FILE)) as handle:
            results = json.load(handle)
        n0 = len(stub.calls)
        again = L.seeds_launch(croot=croot, root_abs=root_abs, root_rel=root_rel, spec=spec, spec_path=sp,
                               spec_sha256=ssha, lattice=lattice, harness_evaluate=stub,
                               ctx=H.CampaignContext(croot, sp, ssha, spec, log=lambda m: None), require_git=False,
                               log=lambda m: None)
        cache_ok = {}
        for arm in L.MODELS:
            ok, cinfo, cache = L.load_seed_cache(root_abs, root_rel, arm, lattice, g0=g0 if arm == 'N' else None,
                                                 require_git=False)
            cache_ok[arm] = {'ok': ok, 'n_entries': len(cache), 'reason': cinfo.get('reason')}
        expected_launches = -(-len(all_labels) // conc)
        r = {'labels': all_labels, 'concurrency': conc, 'n_launches': len(outs),
             'statuses': [o['status'] for o in outs],
             'n_launches_equals_ceil_n_over_concurrency': len(outs) == expected_launches,
             'every_launch_one_batch_of_the_spec_concurrency': [len(o['batches'][0]) for o in outs]
             == [min(conc, len(all_labels) - conc * i) for i in range(expected_launches)],
             'points_bytes_equal_unsplit': _ser(ref_points) == _ser(results['points']),
             'evals_tree_bytes_equal_unsplit': ref_tree == tree_digest(croot, 'evals'),
             'stub_labels_equal_unsplit_in_order': [lab for c in ref_calls for lab in c['labels']]
             == [lab for c in split_calls for lab in c['labels']],
             'results_aliases_from_the_spec': results['aliases'] == spec['extra']['aliases'],
             'final_terminated': outs[-1]['status'] == 'terminated',
             'launch_after_completion_refused': again['status'] == 'refused' and len(stub.calls) == n0,
             'seed_cache_loader_accepts_the_split_results_both_arms': (
                 all(v['ok'] for v in cache_ok.values()) if 'x0_A_2x' in all_labels else
                 all(v['ok'] or 'expected exactly one x = 0 alias' in (v['reason'] or '') for v in cache_ok.values())),
             'cache_loader': cache_ok}
        r['ok'] = all(v for k, v in r.items() if isinstance(v, bool))
        item1_seeds[name] = r
        runs_out[f'item1_seeds_{name}'] = {'ref_points': ref_points, 'outcomes': outs,
                                           'launch_records': read_launch_records(croot)}
        _log(f"  seeds {name}: {len(outs)} launches; ok={r['ok']}")
    res['item1_seeds_split'] = item1_seeds
    checks['item1_seeds'] = {f'{n}:{k}': v for n, r in item1_seeds.items() for k, v in r.items()
                             if isinstance(v, bool)}

    # ---- 1b. the search: split at every batch / poll boundary == the in-process loop ----
    _log('item 1b: search phases, split at every batch / evaluating-poll boundary vs the in-process loop ...')
    key_a = L.make_arm_key_of(lattice, 'A')
    seeds = {s: z for s, z, _n, _y, _t in L.SEEDS}
    z0 = seeds['x0']

    def seed_q_fav(favourite, q_fav, q_other):
        """seed Q: the favourite seed (both models) at q_fav, every other seed at q_other (the favourite is then the
        initial incumbent: q_other - q_fav exceeds every I(x) difference of the seeds, 992,268 EUR at most)."""
        return lambda lab: q_fav if lab.startswith(favourite + '_') else q_other

    # D1: the 7: 0.5/2 incumbent; first poll advanced (k = 2, skipped [0, 1]); the stub then improves along a chain
    # of storage points so that further polls are evaluated after the advanced one
    def design_chain(z_start, k_start, n_steps):
        chain, z, k = [], z_start, k_start
        seen = {z_start}
        for _ in range(n_steps):
            for kk in range(k, k + L.ADVANCE_WINDOW):
                rec, pts = G.poll_points(lattice, key_a, z, kk, sigma_q)
                if pts is not None:
                    break
            nxt = next((p for p in sorted(pts, key=lattice.label) if p not in seen and lattice.has_storage(p)), None)
            if nxt is None:
                break
            chain.append({'from': lattice.label(z), 'polled_k': kk, 'to': lattice.label(nxt)})
            seen.update(pts)
            z, k = nxt, kk + 1
        return chain
    d1_chain = design_chain(seeds['n7_05_2'], 0, 2)
    d1_q = {c['to']: Q0 + 1.0e6 - 3.0e6 * (i + 1) for i, c in enumerate(d1_chain)}
    # A7 overall design (as W183g (b) item 3 b): x0 -> p1 -> p2, one barrier per poll
    pts0 = G.poll_points(lattice, key_a, z0, 0, sigma_q)[1]
    p1 = next(z for z in pts0 if lattice.label(z) == 'y2030__n7_p0.25_e0.5__n9_p0.25_e0.5')
    pts1 = G.poll_points(lattice, key_a, p1, 1, sigma_q)[1]
    p2 = next(z for z in pts1 if z not in pts0 and z != z0 and (G.poll_points(lattice, key_a, z, 2, sigma_q)[1] or [])
              and G.poll_points(lattice, key_a, z, 2, sigma_q)[0]['completion_triggered'] is False)
    pts2 = G.poll_points(lattice, key_a, p2, 2, sigma_q)[1]
    b0 = next(z for z in pts0 if z not in (p1,))
    b1 = next(z for z in pts1 if z not in pts0 and z not in (p2, z0))
    b2 = next(z for z in pts2 if z not in pts0 and z not in pts1 and z not in (z0, p1))
    a7o_q = {lattice.label(p1): Q0 - 3.0e6, lattice.label(p2): Q0 - 6.0e6}
    a7o_barrier = {lattice.label(b) for b in (b0, b1, b2)}
    # move to storage + advance (W183g (b) item 3 c)
    p_st = next(z for z in pts0 if lattice.label(z) == 'y2030__n9_p0.25_e0.5')
    pts_k2 = G.poll_points(lattice, key_a, p_st, 2, sigma_q)[1]
    p_next = next(z for z in pts_k2 if z != z0 and z not in pts0 and lattice.has_storage(z))
    mv_q = {lattice.label(p_st): Q0 - 3.0e6, lattice.label(p_next): Q0 - 6.0e6}
    all_domain = {lattice.label(z) for z in lattice.domain()}
    scen = [
        # the chain improves by 3e6 per step from the incumbent 7: 0.5/2 (Q0 + 1e6); every other point Q0 + 5e6
        Scenario('advance_chain_A_c1', 'A', 1, seed_q_fav('n7_05_2', Q0 + 1.0e6, Q0 + 3.0e6),
                 lambda lab: d1_q.get(lattice_label_of(lab), Q0 + 5.0e6)),
        Scenario('A7_per_poll_N_alias_c1', 'N', 1, seed_q_fav('x0', Q0, Q0 + 2.0e6), lambda lab: Q0,
                 barrier={f'{lab}_N_{L.FLEX_M:g}x' for lab in all_domain - {'x0'}}),
        # as W183g (b) item 3: the incumbent x = 0 at Q0 + 1e6, every undesigned point Q0 + 1e6 (never better)
        Scenario('A7_overall_A_c1', 'A', 1, seed_q_fav('x0', Q0 + 1.0e6, Q0 + 2.0e6),
                 lambda lab: a7o_q.get(lattice_label_of(lab), Q0 + 1.0e6),
                 barrier={f'{lab}_A_{L.FLEX_M:g}x' for lab in a7o_barrier}),
        Scenario('move_to_storage_advance_A_c4', 'A', 4, seed_q_fav('x0', Q0 + 1.0e6, Q0 + 2.0e6),
                 lambda lab: mv_q.get(lattice_label_of(lab), Q0 + 1.0e6)),
        # x = 0 at Q0 + 3e6 against q_hash in Q0 +- 2e6: improvements until the budget of 30 is exhausted
        Scenario('budget_q_hash_N_c4', 'N', 4, seed_q_fav('x0', Q0 + 3.0e6, Q0 + 4.0e6),
                 lambda lab: G.q_hash(lattice_label_of(lab))),
    ]
    item1_search, bases = {}, {}
    for sc in scen:
        r, runs, base, stub = run_scenario(gate_scratch, sc, lattice, sigma_q, g0, args.g0_sha256)
        item1_search[sc.name] = r
        runs_out[f'item1_search_{sc.name}'] = runs
        bases[sc.name] = (base, stub, sc)
    item1_search['advance_chain_A_c1']['design'] = {'chain': d1_chain, 'q': d1_q}
    item1_search['A7_overall_A_c1']['design'] = {'p1': lattice.label(p1), 'p2': lattice.label(p2),
                                                 'barriers': sorted(a7o_barrier)}
    item1_search['move_to_storage_advance_A_c4']['design'] = {'p_storage': lattice.label(p_st),
                                                              'p_next': lattice.label(p_next)}
    res['item1_search_split'] = item1_search
    c1 = {}
    for name, r in item1_search.items():
        for unit in ('batch', 'poll'):
            if f'split_{unit}' in r:
                for k, v in r[f'split_{unit}'].items():
                    if isinstance(v, bool) and not k.startswith('a_launch_ended_'):
                        c1[f'{name}:{unit}:{k}'] = v
    adv = item1_search['advance_chain_A_c1']
    c1['advance_chain:batch:a_launch_ended_inside_an_advanced_poll'] = adv['split_batch'][
        'a_launch_ended_inside_an_advanced_poll']
    c1['advance_chain:poll:a_launch_ended_right_after_an_advanced_poll'] = adv['split_poll'][
        'a_launch_ended_right_after_an_advanced_poll']
    c1['advance_chain:first_poll_advanced_k2_skipped_0_1'] = (
        adv['split_poll']['boundaries'][0]['completed_polls'][0][0] == 2
        and adv['split_poll']['boundaries'][0]['completed_polls'][0][2] == [0, 1])
    c1['A7_per_poll:terminates_on_the_barrier_rule'] = item1_search['A7_per_poll_N_alias_c1']['split_batch'][
        'termination'] == 'STOP_FOR_REVIEW_barrier_rule'
    c1['A7_per_poll:initial_incumbent_is_the_x0_alias'] = item1_search['A7_per_poll_N_alias_c1'][
        'initial_incumbent'] == 'x0_N_2x'
    c1['A7_overall:terminates_on_the_barrier_rule'] = item1_search['A7_overall_A_c1']['split_batch'][
        'termination'] == 'STOP_FOR_REVIEW_barrier_rule'
    c1['A7_overall:a_launch_ended_with_a_barrier_counted'] = any(
        (b['n_barrier_new'] or 0) > 0 and not b['terminated']
        for b in item1_search['A7_overall_A_c1']['split_batch']['boundaries'])
    mv = item1_search['move_to_storage_advance_A_c4']
    c1['move_to_storage:a_launch_ended_inside_an_advanced_poll'] = mv['split_batch'][
        'a_launch_ended_inside_an_advanced_poll']
    c1['budget_q_hash:terminates_on_the_budget'] = item1_search['budget_q_hash_N_c4']['split_batch'][
        'termination'] == 'evaluation_budget_exhausted'
    checks['item1_search'] = c1

    # ---- 2 + 3. planted foreign records / tampered state on mid-run roots ----
    _log('items 2-3: planted foreign records and tampered state ...')
    base_d1, stub_d1, _sc = bases['advance_chain_A_c1']
    b_n = bases['budget_q_hash_N_c4'][0]
    other = {'records': read_launch_records(L.phase_root(f"{b_n['root_abs']}__split_batch", b_n['root_rel'],
                                                         'search_N')),
             'root_abs': b_n['root_abs'], 'renamed_root_abs': f"{b_n['root_abs']}__split_batch"}
    base_d1 = dict(base_d1, phase='search_A', arm='A')
    d1_bound = item1_search['advance_chain_A_c1']['split_batch']['boundaries']
    n_mid = next(i + 1 for i, b in enumerate(d1_bound) if (b['n_polls'] or 0) >= 1 and not b['terminated'])
    backup, mid_outs = mid_run(gate_scratch, base_d1, stub_d1, lattice, sigma_q, 'batch', g0, n_mid, 'd1_batch_mid')
    ctxs = (lattice, sigma_q, base_d1, stub_d1, 'batch', g0, backup)
    item2 = foreign_record_cases(ctxs, other)
    item3 = state_tamper_cases(ctxs)
    # the A7-overall root after its first barrier: the barrier-count tamper with a non-zero count
    base_a7, stub_a7, _sc = bases['A7_overall_A_c1']
    base_a7 = dict(base_a7, phase='search_A', arm='A')
    a7_bound = item1_search['A7_overall_A_c1']['split_batch']['boundaries']
    n_a7 = next(i + 1 for i, b in enumerate(a7_bound) if (b['n_barrier_new'] or 0) > 0 and (b['n_polls'] or 0) >= 1
                and not b['terminated'])
    backup_a7, _o = mid_run(gate_scratch, base_a7, stub_a7, lattice, sigma_q, 'batch', g0, n_a7, 'a7o_batch_mid')
    ctxs_a7 = (lattice, sigma_q, base_a7, stub_a7, 'batch', g0, backup_a7)
    item3_a7 = state_tamper_cases(ctxs_a7)
    # the seeds phase: the evaluated count tampered
    s_root_rel, s_root_abs = 'planted_seeds_tamper', os.path.join(gate_scratch, 'seeds', 'tamper')
    os.makedirs(s_root_abs)
    s_stub = StubHarness(lambda lab: seed_q[lab], name='seeds_tamper')
    s_croot, s_sp, s_ssha, s_spec = plant_seeds_spec(s_root_abs, s_root_rel, lattice, labels7, 1, args.g0_sha256)
    for _ in range(3):
        L.seeds_launch(croot=s_croot, root_abs=s_root_abs, root_rel=s_root_rel, spec=s_spec, spec_path=s_sp,
                       spec_sha256=s_ssha, lattice=lattice, harness_evaluate=s_stub,
                       ctx=H.CampaignContext(s_croot, s_sp, s_ssha, s_spec, log=lambda m: None), require_git=False,
                       log=lambda m: None)
    s_backup = os.path.join(gate_scratch, 'backups', 'seeds_tamper')
    shutil.copytree(s_root_abs, s_backup)
    s_base = {'root_abs': s_root_abs, 'root_rel': s_root_rel, 'croot': s_croot, 'spec_path': s_sp, 'spec_sha': s_ssha,
              'spec': s_spec, 'phase': 'seeds', 'arm': None}
    s_ctxs = (lattice, sigma_q, s_base, s_stub, 'batch', g0, s_backup)

    def _seeds_tamper(field, fn):
        def apply(b):
            n, path, lr = _last_lr(b)
            before = copy.deepcopy(lr['state_after'].get(field))
            fn(lr['state_after'])
            _rewrite_json(path, lr)
            return {'field': field, 'before': before, 'after': lr['state_after'].get(field)}
        return apply
    item3_seeds = [
        attempt('seeds: n_evaluated +1', _seeds_tamper('n_evaluated', lambda s: s.__setitem__(
            'n_evaluated', s['n_evaluated'] + 1)), ['n_evaluated_equals_the_last_launch_record'], s_ctxs),
        attempt('seeds: labels_evaluated reordered', _seeds_tamper('labels_evaluated', lambda s: s.__setitem__(
            'labels_evaluated', list(reversed(s['labels_evaluated'])))),
            ['labels_evaluated_equal_the_last_launch_record'], s_ctxs),
        attempt('seeds: n_not_certified +1', _seeds_tamper('n_not_certified', lambda s: s.__setitem__(
            'n_not_certified', s['n_not_certified'] + 1)), ['n_not_certified_equals_the_last_launch_record'], s_ctxs)]
    # positive control: the untampered mid-run roots resume and complete byte-equal to the in-process reference
    pos = {}
    for tag, b, st, bk, sc_name in (('d1', base_d1, stub_d1, backup, 'advance_chain_A_c1'),
                                    ('a7o', base_a7, stub_a7, backup_a7, 'A7_overall_A_c1')):
        if os.path.exists(b['root_abs']):
            shutil.rmtree(b['root_abs'])
        shutil.copytree(bk, b['root_abs'])
        outs, _calls = split_search(lattice, sigma_q, b['root_abs'], b['root_rel'], b['croot'], b['spec_path'],
                                    b['spec_sha'], b['spec'], st, 'batch', g0)
        ref_out = runs_out[f'item1_search_{sc_name}']['reference']['out']
        with open(os.path.join(b['croot'], L.RESULTS_FILE)) as handle:
            results = json.load(handle)
        pos[tag] = {'resumed_status_first': outs[0]['status'], 'n_launches_after_restore': len(outs),
                    'final_terminated': outs[-1]['status'] == 'terminated',
                    'return_dict_bytes_equal_reference': _ser(ref_out) == _ser(outs[-1].get('result')),
                    'points_bytes_equal_reference': _ser(runs_out[f'item1_search_{sc_name}']['reference']['points'])
                    == _ser(results['points'])}
        pos[tag]['ok'] = pos[tag]['resumed_status_first'] != 'refused' and all(
            v for v in pos[tag].values() if isinstance(v, bool))
    res['item2_foreign_records'] = {'mid_run': {'scenario': 'advance_chain_A_c1', 'unit': 'batch',
                                                'n_launches_before': n_mid,
                                                'states': [o['state_after'] for o in mid_outs]}, 'cases': item2}
    res['item3_state_tamper'] = {'advance_chain_mid_run': item3, 'A7_overall_mid_run_after_a_barrier': {
        'n_launches_before': n_a7, 'cases': item3_a7}, 'seeds_mid_run': item3_seeds, 'positive_control': pos}
    checks['item2'] = {f"{c['case']}": c['ok'] for c in item2}
    checks['item3'] = dict({f"advance_chain:{c['case']}": c['ok'] for c in item3},
                           **{f"A7_overall:{c['case']}": c['ok'] for c in item3_a7},
                           **{c['case']: c['ok'] for c in item3_seeds},
                           **{f'positive_control:{k}': v['ok'] for k, v in pos.items()})
    checks['item3']['A7_overall:the_tampered_barrier_count_was_nonzero'] = any(
        c['detail'].get('before', 0) > 0 for c in item3_a7 if c['case'].startswith('barrier count'))

    # ---- 4. the W183g (b) gate checks on v3 ----
    _log('item 4: the W183g (b) gate checks on v3 ...')
    c4, d4, r4 = w183g_b_checks_on_v3(lattice, sigma_q, table, w183g_b_out)
    res['item4_w183g_b_checks_on_v3'] = d4
    checks['item4'] = c4
    runs_out.update(r4)
    _log(f"source diff: {d4['source_diff']['counts_by_side_and_category']}; uncategorised "
         f"{len(d4['source_diff']['uncategorised_lines'])}")

    # ---- 5. the v1/v2 dry-run assertions on v3; CLI flag rules; the 'resume' root rule ----
    _log('item 5: the launcher dry run in-process ...')
    sub_abs = os.path.join(out_abs, LAUNCHER_DRY_RUN_SUBDIR)
    os.makedirs(sub_abs)
    sub_rel, sub_abs = L.resolve_root(sub_abs)
    dargs = SimpleNamespace(concurrency=1, g0_artifact=args.g0_artifact, g0_sha256=args.g0_sha256,
                            scratch=os.path.join(scratch, 'launcher_dry_run_scratch'),
                            allow_uncommitted=args.allow_uncommitted)
    code = L.dry_run(dargs, sub_rel, sub_abs)
    with open(os.path.join(sub_abs, L.DRY_RUN_FILE)) as handle:
        dr = json.load(handle)
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in L.MODELS}
    g0_ok, g0_info = L.g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True)
    roots = {
        'batch_root_accepts_the_four_at_4': not L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds',
                                                                         list(L.BATCH_SEEDS), 4, labels7),
        'batch_root_refuses_concurrency_1': bool(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds',
                                                                          list(L.BATCH_SEEDS), 1, labels7)),
        'batch_root_refuses_all_seven': bool(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds', labels7, 4,
                                                                      labels7)),
        'batch_root_refuses_search_phases': all(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, p, None, 4, labels7)
                                                for p in ('search_N', 'search_A')),
        'e1_root_accepts_all_seven': not L.declared_root_refusals(L.E1_ROOT_REL, 'seeds', labels7, 1, labels7),
        'e1_root_refuses_a_subset': bool(L.declared_root_refusals(L.E1_ROOT_REL, 'seeds', list(L.BATCH_SEEDS), 4,
                                                                  labels7)),
        'e1_root_accepts_search_phases': all(not L.declared_root_refusals(L.E1_ROOT_REL, p, None, 1, labels7)
                                             for p in ('search_N', 'search_A')),
        'undeclared_root_refused': bool(L.declared_root_refusals(sub_rel, 'seeds', labels7, 1, labels7)),
        'e1_root_declares_cross_under_e1_2x': 'cross' in L.DECLARED_ROOTS[L.E1_ROOT_REL]['phases']
        and 'cross' not in L.PHASES,
        'would_be_spec_records_declared_roots_and_b_plus_rule': (
            dr['would_be_seeds_spec']['content']['extra'].get('declared_roots') == json.loads(_ser(L.DECLARED_ROOTS))
            and dr['would_be_seeds_spec']['content']['extra']['poll_design'].get('b_plus_rule') == L.B_PLUS_RULE),
        'advance_window_10': L.ADVANCE_WINDOW == 10}
    # the 'resume' root rule on planted inventories (pure)
    rr = {}
    inv_complete = L.root_inventory(f"{bases['A7_overall_A_c1'][0]['root_abs']}__split_poll",
                                    bases['A7_overall_A_c1'][0]['root_rel'])
    inv_mid = L.root_inventory(backup, base_d1['root_rel'])
    inv_spec_only = L.root_inventory(base_d1['template'], base_d1['root_rel'])
    rr['resume_refuses_a_complete_phase'] = bool(L.root_refusals(inv_complete, 'resume', 'search_A'))
    rr['resume_accepts_a_mid_run_phase'] = not L.root_refusals(inv_mid, 'resume', 'search_A')
    rr['run_refuses_a_mid_run_phase'] = bool(L.root_refusals(inv_mid, 'run', 'search_A'))
    rr['run_accepts_the_spec_only_phase'] = not L.root_refusals(inv_spec_only, 'run', 'search_A')
    # the CLI flag rules (in-process; parser errors exit before anything runs)
    cli = {}
    for tag, argv in (('resumable_with_dry_run', ['--dry-run', '--resumable', '--root', sub_rel, '--concurrency', '1',
                                                  '--g0-artifact', 'x', '--g0-sha256', 'y', '--scratch', 'z']),
                      ('launch_unit_without_resumable', ['--run', '--launch-unit', 'poll', '--phase', 'search_A',
                                                         '--root', sub_rel, '--spec-sha256', 'x', '--g0-artifact', 'x',
                                                         '--g0-sha256', 'y']),
                      ('unknown_launch_unit', ['--run', '--resumable', '--launch-unit', 'cycle', '--phase', 'search_A',
                                               '--root', sub_rel, '--spec-sha256', 'x', '--g0-artifact', 'x',
                                               '--g0-sha256', 'y'])):
        saved = sys.argv
        sys.argv = [L.SCRIPT_NAME] + argv
        try:
            L.main()
            cli[tag] = {'exited': None}
        except SystemExit as error:
            cli[tag] = {'exited': error.code}
        finally:
            sys.argv = saved
    cli_ok = {f'cli:{k}_parser_error_exit_2': v['exited'] == 2 for k, v in cli.items()}
    cells = {(c['row'], c['cell'][:12]): c for c in dr['committed_cell_comparison']['cells']}
    cell35 = next(c for c in dr['committed_cell_comparison']['cells'] if c['row'] == 35)
    cell40 = next(c for c in dr['committed_cell_comparison']['cells'] if c['row'] == 40
                  and c['cell'].startswith('ref:5ca4f86c (issuing run'))
    res['item5'] = {'launcher_dry_run_exit_code': code, 'launcher_dry_run_root': sub_rel,
                    'launcher_dry_run_file': os.path.join(sub_rel, L.DRY_RUN_FILE),
                    'launcher_dry_run_sha256': _sha(os.path.join(sub_rel, L.DRY_RUN_FILE)),
                    'launcher_dry_run_failing_assertions': dr['failing_assertions'],
                    'seed_key_summary': dr['seed_key_summary'], 'prediction': dr['prediction'], 'alias': dr['alias'],
                    'g0_require_current_production': {'accepted': g0_ok, **g0_info},
                    'committed_cells': {f'{k[0]}:{k[1]}': {kk: v.get(kk) for kk in (
                        'committed_base_key', 'launcher_key_2x', 'base_key_equals_launcher_key',
                        'differing_key_inputs')} for k, v in cells.items()},
                    'declared_roots': roots, 'resume_root_rule': rr, 'cli': cli}
    checks['item5'] = {
        'launcher_dry_run_exit_0_all_assertions_hold': code == 0 and dr['all_assertions_hold'] is True,
        'seven_keys_held': dr['prediction']['held'] is True and dr['seed_key_summary']['n_distinct_keys_to_evaluate']
        == 7,
        'one_alias_g0_final_head': (dr['seed_key_summary']['n_aliases'] == 1 and dr['alias']['g0_accepted'] is True
                                    and dr['alias']['g0_artifact_sha256'] == G.G0_FINAL['sha256']
                                    and dr['g0']['artifact_production']['equals_current_production'] is True),
        'g0_accepted_with_require_current_production': g0_ok,
        'base_key_equal_row_35': cell35.get('base_key_equals_launcher_key') is True,
        'base_key_equal_row_40': cell40.get('base_key_equals_launcher_key') is True,
        'root_scoped_cache': all(dr['assertions']['cache_scope'].values()),
        'pins': all(dr['assertions']['pins_recorded'].values()),
        'preflight_recorded': all(dr['assertions']['concurrency_and_preflight'].values()),
        'rule_eleven_e1_loop_checks_8': all(v for k, v in dr['rule_eleven']['checks'].items()
                                            if k.startswith('e1_loop'))
        and sum(k.startswith('e1_loop') for k in dr['rule_eleven']['checks']) == 8,
        'rule_eleven_resume_state_probe_fields': dr['rule_eleven']['checks'].get('resume_state_probe_fields') is True,
        'first_poll_table_in_dry_run_equals_w183b': dr['first_poll_if_each_seed_were_the_incumbent'] == table,
        **{f'roots:{k}': v for k, v in roots.items()}, **{f'resume_root_rule:{k}': v for k, v in rr.items()},
        **cli_ok}

    # ---- the per-launch time bound (informative) ----
    res['per_launch_time_bound'] = time_bound()

    # ---- guards, outputs ----
    g = L.guards_report()
    gate_failures = GATE_GUARD.verify(0)
    res['guards'] = dict(g, gate_guard={'counts': dict(GATE_GUARD.counts), 'verify_0_failures': gate_failures},
                         n_guards_verified=len(L.all_guards()))
    checks['guards'] = {'all_zero': g['all_zero'] and not gate_failures,
                        'gate_guard_among_verified': any(gd is GATE_GUARD for _n, gd in L.all_guards())}
    res['checks'] = checks
    failing = sorted(f'{grp}.{k}' for grp, cs in checks.items() for k, v in cs.items() if v is not True)
    res['n_checks'] = sum(len(cs) for cs in checks.values())
    res['n_checks_by_item'] = {grp: len(cs) for grp, cs in checks.items()}
    res['failing_checks'] = failing
    res['GATE_PASS'] = not failing
    shutil.rmtree(scratch)
    res['scratch_removed'] = not os.path.exists(scratch)
    res['finished_utc'] = L._utc()
    res['wall_clock_s'] = time.time() - started
    runs_path = os.path.join(out_abs, OUT_RUNS_FILE)
    with open(runs_path, 'xb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as gz:
            gz.write(GRIO.dumps(runs_out, sort_keys=True).encode())
    res['loop_outputs_file'] = {'path': os.path.join(out_rel, OUT_RUNS_FILE), 'sha256': _sha(runs_path),
                                'content': ('every reference / split run: return dicts, points, stub calls, launch '
                                            'outcomes and launch records; the W183g (b) paired runs on v3')}
    out_path = os.path.join(out_abs, OUT_FILE)
    sha = L._write_once_json(out_path, res)
    manifest = {os.path.join(out_rel, OUT_FILE): sha, os.path.join(out_rel, OUT_RUNS_FILE): _sha(runs_path),
                os.path.join(sub_rel, L.DRY_RUN_FILE): _sha(os.path.join(sub_rel, L.DRY_RUN_FILE)),
                os.path.join(sub_rel, L.DRY_RUN_MANIFEST): _sha(os.path.join(sub_rel, L.DRY_RUN_MANIFEST)),
                L.SCRIPT_NAME: _sha(L.SCRIPT_NAME), SCRIPT_NAME: _sha(SCRIPT_NAME),
                W183G_B_GATE['path']: _sha(W183G_B_GATE['path']), W183G_B_OUT['path']: _sha(W183G_B_OUT['path']),
                G.W183B_DRYRUN['path']: _sha(G.W183B_DRYRUN['path']), G.G0_FINAL['path']: _sha(G.G0_FINAL['path']),
                **{m: _sha(m) for m in UNCHANGED_MODULES}}
    L._write_once_json(os.path.join(out_abs, OUT_MANIFEST), manifest)
    _log(f'checks by item: {res["n_checks_by_item"]}; failing: {failing}')
    _log(f"GATE_PASS={res['GATE_PASS']} ({res['n_checks'] - len(failing)}/{res['n_checks']}); wrote "
         f"{os.path.join(out_rel, OUT_FILE)} sha256 {sha}")
    _log(f"guards: all_zero={g['all_zero']} gate_guard={GATE_GUARD.counts} pickle={g['pickle']['counts']} "
         f"build={g['model_build_tripwire']}")
    for _n, guard in reversed(L.all_guards()):
        guard.uninstall()
    return 0 if res['GATE_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
