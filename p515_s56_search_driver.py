"""P5.15 S56 EV search -- the OPPORTUNISTIC SEARCH DRIVER (freeze, dry run, one launch per --run, resume).

Authority: S56_SEARCH_DRIVER_DESIGN.md (commit 2c15db47) -- implemented as written; TASKS.md "Working mode and EV
priority" (critical path step 2) and "Search policy review (O1-O15)"; S56_PLAN_v4_1.md sections 3-4. Every DECISION
is `p515_s56_search_policy` (P); this module supplies N(x), evaluates exactly what `P.step` asks for, stores the
records and proves, on every launch, that the stored history re-derives. Gate: `p515_s56_search_gate.py`.

REUSED BY IMPORT: `p516_lattice` (the lattice, `GeneralLattice.key`, I(x)); `p515_s56_neighbourhood.s56_neighbourhood`
(N(x)); `p515_s44_campaign_harness` (`evaluate`, the campaign lock, write-once eval dirs, heartbeat tags, stub
children; harness oracle only); `p515_s55_e1_launcher.memory_preflight` (harness oracle only). NOT reused: E1's
Halton poll and best-of-poll acceptance, `PB.classify` / `PB.resolution`, `resolution_by_bound`.

THE SEARCH ROOT (write-once files; nothing is ever deleted)
  search_spec_<sha8>.json                 frozen by --freeze: arm, configuration, start, N_max, K_EV, R, e floor,
                                          block 5, cap 80, class rank, instance, oracle, cache sources, code hashes
  launches/launch_<NNN>/                  ONE launch = ONE action of `P.step` (an exit block of <= 5, or one
                                          confirmation batch): its own harness campaign root (frozen harness spec,
                                          evals/<key16>_<label>/) -- or, for the synthetic oracle, evals/ only
  launches/launch_<NNN>.json              the launch record: the action, the records it produced (z, depth, eval key,
                                          eval dir, record file sha256, summary), the state returned with the action,
                                          previous_launch_manifest_sha256 (the hash chain), times
  launches/launch_<NNN>_manifest_sha256.json   sha256 of every file the launch created + the launch record
  interrupted/interrupted_<MMM>.json + interrupted/aside_<MMM>/   an interrupted launch: its files moved aside,
                                          its evaluations listed with their hours (never counted)
  stale_locks/lock_<MMM>.json             a stale campaign lock taken over explicitly (moved, not deleted)
  search_state.json                       DERIVED VIEW: the replayed state at the resume point (rewritten each launch)
  search_results.json + search_results_manifest_sha256.json   written once, when `P.step` terminates
THE TRUTH is the launch chain. Every --run first REPLAYS it: initial state from the spec; for launch k,
`P.step(state, records_<k)` must return exactly launch k's action and state; launch k's records are then re-derived
from their eval dirs (record file hash, summary, eval key recomputed from z and depth) and added. A mismatch refuses
the launch (exit 2) and writes nothing. The view must equal the replayed state of the launch it names.
RESUME AFTER A KILL: an incomplete launch (a launch dir or record without its manifest; results without their
manifest) is recorded in interrupted_<MMM>.json -- every eval dir with its start, last write and hours -- and moved
aside; its evaluations never count (`P.step` re-issues the pending action). A stale campaign lock is taken over only
with --take-over-stale-lock, only when its pid is dead and it names this search; it is moved to stale_locks/.
EXIT CODES: 0 terminated normally (complete poll, budget exhausted); 5 terminated for review (harness_error,
stop_for_review_offset, stop_for_review_not_settled, start not certified); 4 launch done, more work remains (re-issue the same command); 2 refused
(nothing written); 1 crash. ONE launch = one action, attached, both streams captured, never detached; for unattended
progress the author re-issues in a foreground loop under `caffeinate -i`. One search at a time per checkout (the
harness campaign lock).

ORACLES (--oracle at --freeze; recorded in the spec):
  synthetic      in-process synthetic landscape (dry runs and the gate); writes one record file per evaluation
  stub_children  the harness `evaluate` with stub children (gate only; values from the synthetic landscape)
  harness        real evaluations through the oracle mapping (`map_record`; integration build). The mapping parameters
                 are FINAL (integration rulings 1-7, S56_SEARCH_DRIVER_DESIGN.md, 601775c0); a real freeze refuses while
                 any is provisional or memory_per_slot_gib is undeclared; a real launch runs `real_launch_preflight`
                 (harness preconditions, solver path, memory rule) before anything is written.
CACHE ROOTS (--cache-root, repeatable, at --freeze; harness oracles): every eval dir whose record carries the key this
search would request is indexed in the spec (`build_cache_index`) and re-verified and mapped at every launch
(`load_cache`); START POOL (--start best_of_pool --start-pool FILE): x0, the reference unit and the declared members;
the start is the certified member with the lowest cached exit F (`best_start`, plan v4.1 section 4).
PILOT (--pilot-freeze / --pilot-run / --pilot-evaluate; S56_SEARCH_DRIVER_DESIGN.md "Pilot protocol"): see
`pilot_spec`. A real pilot freeze needs --predictions (the recorded predictions, embedded verbatim).
`--pilot-run --preflight-only --out <new dir>` runs, at zero solves, exactly the checks a real launch of the next block
runs before it writes anything (lock, chain, `real_launch_preflight`), writes the record to --out, releases the lock
and launches nothing.

--dry-run (zero solves, guard armed): the full search against the synthetic oracle in process, for a named start;
writes nothing but its own output directory (a NEW directory).

THE 2x PHASE (TASKS.md "Author decisions on report (i)", step 5a; section "THE 2x PHASE" below; decisions in
`P.phase_step`): --phase-freeze (real: --oracle harness --predictions FILE; test-only: synthetic / stub_children with
--phase-test-config) and --phase-run (one launch = one block of <= 5; exit 4 more work, 0 complete, 5 review, 6 the
120 h ceiling, 2 refused, 1 crash; --preflight-only --out <new dir> at zero solves). The author's loop re-issues
--phase-run while the exit code is 4. Spec v2 (TASKS 5a, Planner decisions R1, R4, P1, P2): --unlaunched-predecessor
ROOT@SHA (excluded from the collision scan once verified un-launched), --continuation-of ROOT@SHA (a stopped
predecessor with launches as a cache root; with --ceiling-h / --ceiling-note and --no-pilot-exit-views), the
code-drift refusal at every --phase-run, .DS_Store ignored; `phase_single_launch_command` (the first-block hold) and
`s56_phase_check.py` (the launch-record check).
THE 1x PHASE (TASKS.md step 5d, 2026-11-26): --phase-freeze --phase-flex 1.0 --no-pilot-exit-views -- the same design at
flexibility price 1.0, ceiling 60 h, predictions Q1-Q9, no cache source (section comment at PHASE_FLEX_LEVELS).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

import gate_result_io as GRIO  # noqa: E402  (stdlib only)
import p515_s56_search_policy as P  # noqa: E402  (stdlib only)

SCRIPT_NAME = os.path.basename(__file__)
SEARCH_SPEC_SCHEMA = 'p515_s56_search_spec_v1'
LAUNCH_RECORD_SCHEMA = 'p515_s56_search_launch_record_v1'
RESULTS_SCHEMA = 'p515_s56_search_results_v1'
SYNTHETIC_RECORD_SCHEMA = 'p515_s56_synthetic_record_v1'
CASE_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3.json')
DATA_DIR_REL = os.path.join('data', 'SRP1')
RESULTS_PARENT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56')
DOMAIN_SIZE = 37042            # audit J / diagnostic 1 (37,041 plans + x0); re-derived by the gate (item 1)
ORACLES = ('synthetic', 'stub_children', 'harness')
EXIT_TERMINATED, EXIT_FAILED, EXIT_REFUSED, EXIT_SUSPENDED, EXIT_REVIEW = 0, 1, 2, 4, 5
REVIEW_TERMINATIONS = ('harness_error', 'stop_for_review_offset', 'start_incumbent_not_certified',
                       'stop_for_review_not_settled')   # the last: integration ruling 5 (601775c0)
LAUNCH_FMT, MANIFEST_FMT = 'launch_{:03d}', 'launch_{:03d}_manifest_sha256.json'
RESULTS_FILE, RESULTS_MANIFEST = 'search_results.json', 'search_results_manifest_sha256.json'
STATE_VIEW = 'search_state.json'
KILL_ENV = 'S56_SEARCH_TEST_KILL'          # gate only: '<transition>:<launch index>[:<delay s>]', test-only specs
KILL_TRANSITIONS = ('after_lock', 'after_launch_dir', 'during_evaluate', 'after_evaluate', 'after_launch_record',
                    'after_manifest', 'after_state_view', 'after_results')
NAMED_STARTS = {  # the named points of diagnostic 1 (p515_s56_neighbourhood)
    'x0': {},
    'reference_unit': {(7, 2025): (1, 2)},
    'plan_2030': {(5, 2030): (1, 1), (7, 2030): (4, 7)},
}
AUTHORITY = ['S56_SEARCH_DRIVER_DESIGN.md (2c15db47)', 'TASKS.md: Working mode and EV priority; Search policy review',
             'S56_PLAN_v4_1.md sections 3-4']


class Refused(Exception):
    def __init__(self, reasons):
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__('; '.join(self.reasons))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56-SEARCH] {msg}', flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_once_json(path, obj):
    """Write-once JSON through the shared gate writer; returns the file's sha256."""
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True)
    return sha256_file(path)


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return sha256_file(path)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


# Finder metadata files: never counted, manifested or treated as foreign wherever the driver lists files under a root
# (phase spec v2, Planner decision P2). Exact name only.
IGNORED_FILE_NAMES = ('.DS_Store',)


def files_under(path):
    out = []
    for directory, _dirs, files in os.walk(path):
        out += [os.path.join(directory, f) for f in files if f not in IGNORED_FILE_NAMES]
    return sorted(out)


# ======================================================================================================================
#  the lattice and N(x)
# ======================================================================================================================
class LatticeEnv:
    """The `env` of `P.step`: the lattice and N(x) = `s56_neighbourhood(lattice, z)['points']` (memoized)."""

    def __init__(self, lattice, memoize=True):
        import p515_s56_neighbourhood as NB  # local: p516_lattice imports the harness module (stdlib only)
        self.lattice = lattice
        self.memoize = memoize
        self._nb = NB.s56_neighbourhood
        self._classes = tuple(NB.CLASS_ORDER)
        self._memo = {}

    def neighbourhood_points(self, z):
        z = tuple(z)
        if z not in self._memo:
            nb = self._nb(self.lattice, z, keep_points=True)
            # N(x) = the points of the classes in CLASS_ORDER (R3, d, e, f); `points` also carries the informational
            # class e_opp (opposite-sign diagonals), which is NOT in N(x) (nb['size'] excludes it)
            pts = [{'class': p['class'], 'part': p['part'], 'move': p['move'], 'z': tuple(p['z'])}
                   for p in nb['points'] if p['class'] in self._classes]
            if len(pts) != nb['size']:
                raise Refused(f'N({list(z)}): {len(pts)} points of the N(x) classes != size {nb["size"]}')
            if not self.memoize:
                return pts
            self._memo[z] = pts
        return [dict(p) for p in self._memo[z]]


def build_lattice(data_root):
    import p516_lattice as GL
    case_path = os.path.join(data_root, CASE_REL)
    with open(case_path) as handle:
        case = json.load(handle)
    inst = GL.load_instance_from_case_dict(case, os.path.join(data_root, DATA_DIR_REL),
                                           case_source={'path': CASE_REL, 'sha256': sha256_file(case_path)})
    lat = GL.GeneralLattice(inst)
    if lat.nodes != (5, 7, 9) or lat.years != (2025, 2030, 2035):
        raise Refused(f'instance nodes/years {lat.nodes}/{lat.years} != (5, 7, 9)/(2025, 2030, 2035)')
    return lat, {'case': CASE_REL, 'case_sha256': sha256_file(case_path), 'lattice': inst.describe()}


def enumerate_domain(lat):
    """The staged admissible domain (37,042 plans incl. x0 on the 3 x 3 instance): diagnostic 1's enumeration with the
    LATTICE rule as the decider (diagnostic 1 used production's check and recorded 0 disagreements with the lattice
    rule on every generated point)."""
    import bisect
    from itertools import product
    import p516_lattice as GL
    ze_cap = int(round(lat.inst.max_capacity_mwh / GL.GRANULE_E_MWH))
    opts = [(0, 0)] + [(p, e) for e in range(1, ze_cap + 1) for p in range(1, ze_cap + 1) if p <= e <= 2 * p]
    unit = {y: (sum(a for a, _b in lat._coef[y]) * GL.GRANULE_P_MVA, sum(b for _a, b in lat._coef[y]) * GL.GRANULE_E_MWH)
            for y in lat.years}
    budget = lat.inst.budget_eur
    per_node = []
    for combo in product(opts, repeat=len(lat.years)):
        if sum(e for _p, e in combo) > ze_cap:
            continue
        c = sum(unit[y][0] * p + unit[y][1] * e for (p, e), y in zip(combo, lat.years))
        if c <= budget + 1.0:
            per_node.append((c, combo))
    per_node.sort()
    costs = [c for c, _ in per_node]
    domain = []
    for c1, t1 in per_node:
        for c2, t2 in per_node:
            if c1 + c2 > budget + 1.0:
                break
            j_end = bisect.bisect_right(costs, budget + 1.0 - c1 - c2)
            for _c3, t3 in per_node[:j_end]:
                z = []
                for trip in (t1, t2, t3):
                    for p, e in trip:
                        z += [p, e]
                z = tuple(z)
                if lat.admissible(z):
                    domain.append(z)
    return sorted(domain)


def named_start(lattice, name):
    z = lattice.x0()
    for (n, y), (zp, ze) in NAMED_STARTS[name].items():
        z = lattice.with_pair(z, n, y, zp, ze)
    return z


# ======================================================================================================================
#  oracles
# ======================================================================================================================
class SyntheticLandscape:
    """A deterministic synthetic F(z) for dry runs and the gate (NOT a model of the system): a bowl around `target`
    plus hashed noise, hashed bars, ~5 % uncertified exits, settled values within 1 kEUR of the exit, ~10 % of
    settled pairs failing V_pair, ~3 % failing a gate. Everything is a pure function of (seed, z, depth)."""

    def __init__(self, seed, target, start, alpha=30000.0, f0=810.0e6, knobs=None):
        self.seed, self.target, self.start = str(seed), tuple(target), tuple(start)
        self.alpha, self.f0 = float(alpha), float(f0)
        # phase tests only (test-only specs): z-sets forcing a status, and settled offsets; None = the plain landscape
        k = knobs or {}
        zs = (lambda name: {tuple(z) for z in k.get(name, [])})
        self.knobs = knobs
        self.always_ok, self.k_exit_he, self.k_settled_he = zs('always_ok'), zs('exit_harness_error'), \
            zs('settled_harness_error')
        self.k_not_settled, self.k_exit_unc = zs('not_settled'), zs('exit_uncertified')
        self.k_offset = {tuple(z): float(v) for z, v in k.get('settled_offset', [])}
        # phase spec v2 tests: exit summaries carry a read_cycle and settled summaries the exit view of the same run
        # (the real mapping's fields); 'exit_view_mismatch' = [[z, field]] perturbs one field of z's settled exit view
        self.k_views = k.get('exit_views') is True
        self.k_view_mismatch = {tuple(z): str(f) for z, f in k.get('exit_view_mismatch', [])}

    def describe(self):
        out = {'seed': self.seed, 'target': list(self.target), 'start': list(self.start), 'alpha': self.alpha,
               'f0': self.f0, 'text': SyntheticLandscape.__doc__}
        if self.knobs is not None:
            out['knobs'] = self.knobs
        return out

    def _u(self, z, tag):
        h = sha256_text(f'{self.seed}|{tag}|{" ".join(str(v) for v in z)}')
        return int(h[:13], 16) / float(1 << 52)

    def summary(self, z, depth):
        z = tuple(z)
        d2 = sum((a - b) ** 2 for a, b in zip(z, self.target))
        f_exit = self.f0 + self.alpha * d2 + (self._u(z, 'noise') - 0.5) * 6000.0
        if depth == P.DEPTH_EXIT:
            if z in self.k_exit_he:
                return {'depth': P.DEPTH_EXIT, 'status': 'harness_error', 'F': None, 'bar': None,
                        'wall_s': 3600.0 * (1.0 + self._u(z, 'wall'))}
            status = 'certified' if (z == self.start or z in self.always_ok or self._u(z, 'cert') >= 0.05) \
                else 'not_certified'
            if z in self.k_exit_unc:
                status = 'not_certified'
            out = {'depth': P.DEPTH_EXIT, 'status': status, 'F': f_exit,
                   'bar': 3600.0 + self._u(z, 'bar') * 26000.0, 'wall_s': 3600.0 * (1.0 + self._u(z, 'wall'))}
            if self.k_views:
                out['read_cycle'] = self._read_cycle(z)
            return out
        if z in self.k_settled_he:
            return {'depth': P.DEPTH_SETTLED, 'status': 'harness_error', 'F': None, 'bar': None,
                    'wall_s': 7200.0 * (1.0 + self._u(z, 'wall'))}
        ok = z in self.always_ok
        s = 2400.0 if (self._u(z, 'vfail') < 0.10 and not ok) else 200.0 + self._u(z, 'S') * 400.0
        out = {'depth': P.DEPTH_SETTLED, 'status': 'not_settled' if z in self.k_not_settled else 'settled',
               'F': f_exit + (self._u(z, 'off') - 0.5) * 2000.0 + self.k_offset.get(z, 0.0),
               'V': {'S': s, 'PR': self._u(z, 'PR') * 500.0, 'G': self._u(z, 'G') * 300.0,
                     'A': self._u(z, 'A') * 1000.0},
               'gates_pass': ok or self._u(z, 'gate') >= 0.03, 'wall_s': 7200.0 * (1.0 + self._u(z, 'wall'))}
        if self.k_views:
            ex = self.summary(z, P.DEPTH_EXIT)
            view = {'status': ex['status'], 'F': ex['F'], 'bar': ex['bar'], 'read_cycle': ex.get('read_cycle'),
                    'read_at': 'would-be certification cycle (stop disabled; synthetic)'}
            f = self.k_view_mismatch.get(z)
            if f == 'read_cycle':
                view['read_cycle'] = view['read_cycle'] + 1
            elif f in ('F', 'bar'):
                view[f] = math.nextafter(view[f], math.inf)   # one ulp: a bitwise, not a numerical, difference
            elif f == 'absent':
                view = None
            if view is not None:
                out['exit_view'] = view
        return out

    def _read_cycle(self, z):
        return 100 + int(self._u(z, 'read_cycle') * 80.0)


def synthetic_landscape_of(spec, lattice):
    o = spec['oracle']
    return SyntheticLandscape(o['seed'], o['target'], spec['start']['z'], o.get('alpha', 30000.0),
                              o.get('f0', 810.0e6), knobs=o.get('knobs'))


def synthetic_eval_key(lattice, spec, z, depth):
    """Identity of one synthetic evaluation: candidate key (`GeneralLattice.key`) x depth x arm x oracle seed."""
    payload = {'candidate_key': lattice.key(tuple(z)), 'depth': depth, 'arm': spec['arm'],
               'oracle': spec['oracle']['kind'], 'seed': spec['oracle']['seed']}
    return sha256_text(P.canonical_json(payload))


def eval_label(lattice, z, depth, arm):
    return f'{lattice.label(tuple(z))}__{depth}__{arm}'


def _dir_name(key, label):
    out = ''.join(c if (c.isalnum() or c == '_') else '_' for c in label.lower())
    return f'{key[:16]}_{out}'


class SyntheticOracle:
    """In-process: one write-once eval dir per evaluation with `evaluation_record.json` (the synthetic summary)."""
    kind = 'synthetic'

    def __init__(self, spec, lattice):
        self.spec, self.lattice = spec, lattice
        self.land = synthetic_landscape_of(spec, lattice)

    def eval_key(self, z, depth):
        return synthetic_eval_key(self.lattice, self.spec, z, depth)

    def evaluate(self, action, launch_dir, hooks, launch_index=None):
        out = []
        evals = os.path.join(launch_dir, 'evals')
        os.makedirs(evals, exist_ok=True)
        for i, z in enumerate(action['zs']):
            if i == len(action['zs']) - 1:
                hooks.at('during_evaluate')   # gate: deterministic kill before the action's last evaluation
            key = self.eval_key(z, action['depth'])
            ed = os.path.join(evals, _dir_name(key, eval_label(self.lattice, z, action['depth'], self.spec['arm'])))
            os.makedirs(ed)
            started = time.time()
            write_once_json(os.path.join(ed, 'launch.json'), {'started_t': started, 'started_utc': _utc(),
                                                               'pid': os.getpid(), 'z': list(z),
                                                               'depth': action['depth'], 'eval_key': key})
            rec = {'schema': SYNTHETIC_RECORD_SCHEMA, 'synthetic': True, 'z': list(z), 'depth': action['depth'],
                   'eval_key': key, 'candidate_key': self.lattice.key(tuple(z)),
                   'summary': self.land.summary(z, action['depth']), 'ended_t': time.time()}
            write_once_json(os.path.join(ed, 'evaluation_record.json'), rec)
            out.append({'z': list(z), 'depth': action['depth'], 'eval_key': key, 'eval_dir': ed})
        return out

    def verify_launch(self, _launch_dir, _action):
        return []

    def summary_from_dir(self, eval_dir, z, depth):
        rec = load_json(os.path.join(eval_dir, 'evaluation_record.json'))
        if rec.get('schema') != SYNTHETIC_RECORD_SCHEMA or rec.get('z') != list(z) or rec.get('depth') != depth:
            raise Refused(f'synthetic record {eval_dir} does not hold {list(z)} {depth}')
        return rec['summary'], os.path.join(eval_dir, 'evaluation_record.json')


STUB_DELAY_ENV = 'S56_SEARCH_TEST_STUB_DELAYS'   # gate only: '<lo s>:<hi s>:<seed>' per-child random stub sleeps


def staged_plan_of(lattice, z):
    """The harness `s56_staged_investment` option of z: every node-year as [s MVA, e MWh] (zeros included)."""
    import p516_lattice as GL
    return {str(n): {str(y): [GL.GRANULE_P_MVA * lattice.pair(z, n, y)[0] + 0.0,
                              GL.GRANULE_E_MWH * lattice.pair(z, n, y)[1] + 0.0] for y in lattice.years}
            for n in lattice.nodes}


def stub_context(H, launch_dir, spec_path, sha, hspec, launch_index, lock, test_only):
    """A harness CampaignContext whose children run the harness STUB mode (no model import), with per-child random
    sleeps from STUB_DELAY_ENV (test-only specs) and the scratch lock path."""
    raw = os.environ.get(STUB_DELAY_ENV)
    if raw and not test_only:
        raise Refused(f'{STUB_DELAY_ENV} is set but the spec is not test-only')
    lo, hi, seed = (raw.split(':') if raw else ('0.2', '0.2', 'fixed'))
    rng = random.Random(f'{seed}|{launch_index}')

    class _Ctx(H.CampaignContext):
        @property
        def child_extra_args(self):
            return ('--stub-mode', 'ok', '--stub-sleep-s', f'{rng.uniform(float(lo), float(hi)):.3f}',
                    '--stub-alloc-mb', '8', '--lock-path', lock)

        @child_extra_args.setter
        def child_extra_args(self, _value):
            pass
    return _Ctx(launch_dir, spec_path, sha, hspec, log=_log)


def compose_eval_key(H, cfg, opts, cap, ck):
    """The eval key `freeze_campaign_spec` gives an S56 staged entry with options `opts` under configuration `cfg` and
    spec cap `cap` (candidate key `ck`): `evaluation_key`, then the S56 EV-plan, settled-run and pricing-capture wraps,
    then the cap wrap (last). Asserted against every frozen launch spec and by the tests."""
    ekey = H.evaluation_key(
        ck, H.validate_overrides(opts['overrides']) if 'overrides' in opts else
        H.validate_overrides(cfg.get('overrides')),
        case_file_aa=H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration')),
        model_variant=opts.get('model_variant'),
        ess_ageing_baseline=H.validate_ess_ageing_baseline(cfg.get('ess_ageing_baseline')),
        flex_price_multiplier=opts.get('flex_price_multiplier'),
        derived_instance=H.validate_derived_instance(cfg.get('derived_instance')),
        interface_deviation_premium=opts.get('interface_deviation_premium'),
        convergence_depth_tail=H.validate_convergence_depth_tail(cfg.get('convergence_depth_tail')),
        rho0_scale=opts.get('rho0_scale'), nrf_option=opts.get('nrf_option'),
        interface_coupling=opts.get('interface_coupling'))
    ekey = H.s56_evaluation_key(ekey, H.s56_ev_plan_in_key(H.validate_s56_ev_plan(opts)))
    ekey = H.s56_settled_stop_evaluation_key(ekey, opts.get(H.S56_SETTLED_STOP_DISABLED_KEY))
    ekey = H.s56_pricing_capture_evaluation_key(ekey, opts.get(H.S56_PRICING_CAPTURE_KEY))
    return H.s56_cycle_cap_evaluation_key(ekey, int(cap))   # every S56 staged entry: the cap enters, last


class HarnessOracle:
    """The campaign harness: one frozen harness spec per launch (its own campaign root = the launch dir), its entries
    the action's candidates at the action's depth, evaluated by ONE `H.evaluate` call (concurrency from the search
    spec). Every eval key the harness freezes is recomputed here (`eval_key`) and must agree. Stub children
    ('stub_children'): the harness stub mode (no model import), values from the synthetic landscape, attached only to
    a stub record of the same candidate. Real records ('harness'): `map_record` (the oracle mapping), after
    `assert_mapping_capture_paths` at construction; every real launch passes `real_launch_preflight` first."""

    def __init__(self, spec, lattice, lock_path):
        import p515_s44_campaign_harness as H
        self.H, self.spec, self.lattice, self.lock_path = H, spec, lattice, lock_path
        self.kind = spec['oracle']['kind']
        self.stub = self.kind == 'stub_children'
        self.land = synthetic_landscape_of(spec, lattice) if self.stub else None
        self.hz = spec['harness']
        self.mapping = spec.get('mapping') or MAPPING_PARAMETERS
        if not self.stub:  # real records: the mapping's capture paths asserted before anything runs (rule eleven)
            self.mapping_checklist = assert_mapping_capture_paths(H, self.hz)
        self.lock_content = None
        sk = H.S56_SETTLED_STOP_DISABLED_KEY
        if self.hz['by_depth']['settled']['entry_options'].get(sk) is not True or \
                self.hz['by_depth']['exit']['entry_options'].get(sk) is not None:
            raise Refused(f'the settled depth must declare {sk} True and the exit depth must not declare it')
        for depth in P.DEPTHS:
            cfg = self.hz['by_depth'][depth]['configuration']
            case_rel = H.s56_case_rel(H.validate_derived_instance(cfg.get('derived_instance')))
            hl = H.s56_lattice(case_rel)
            same = (hl.nodes, hl.years, hl.dim) == (lattice.nodes, lattice.years, lattice.dim) and \
                P.canonical_json({k: v for k, v in hl.inst.describe().items() if k != 'sources'}) == \
                P.canonical_json({k: v for k, v in lattice.inst.describe().items() if k != 'sources'})
            if not same:
                raise Refused(f'the harness lattice of {case_rel} differs from the driver lattice ({CASE_REL})')

    def entry_options(self, z, depth):
        opts = copy.deepcopy(self.hz['by_depth'][depth]['entry_options'])
        opts[self.H.S56_STAGED_INVESTMENT_KEY] = staged_plan_of(self.lattice, z)
        return opts

    def eval_key(self, z, depth):
        """The harness eval key of (z, depth), composed from the harness's own key functions exactly as
        `freeze_campaign_spec` composes it for an S56 staged entry (asserted against every frozen launch spec)."""
        return compose_eval_key(self.H, self.hz['by_depth'][depth]['configuration'], self.entry_options(z, depth),
                                int(self.hz['by_depth'][depth]['cap']),
                                self.H.candidate_key(self.lattice.canonical_of_z(tuple(z))))

    def _ctx(self, launch_dir, spec_path, sha, hspec, launch_index):
        H = self.H
        lock = self.lock_path
        if not self.stub:  # real children: the default lock path only (the child verifies it), no stub arguments
            if os.path.abspath(lock) != os.path.abspath(H.CAMPAIGN_LOCK_PATH):
                raise Refused(f'real children verify the default campaign lock {H.CAMPAIGN_LOCK_PATH}, not {lock}')
            return H.CampaignContext(launch_dir, spec_path, sha, hspec, log=_log)
        return stub_context(H, launch_dir, spec_path, sha, hspec, launch_index, lock, self.spec.get('test_only'))

    def evaluate(self, action, launch_dir, hooks, launch_index=None):
        H = self.H
        depth = action['depth']
        bd = self.hz['by_depth'][depth]
        labels = [eval_label(self.lattice, z, depth, self.spec['arm']) for z in action['zs']]
        cands = [(lab, None, self.entry_options(z, depth)) for lab, z in zip(labels, action['zs'])]
        cfg = dict(copy.deepcopy(bd['configuration']), name=f"{self.spec['search_id']} launch {launch_index} {depth}")
        spec_path, sha, hspec = H.freeze_campaign_spec(
            launch_dir, f"{self.spec['search_id']}_l{launch_index:03d}", cands, cfg, bd['cap'],
            self.hz['concurrency'], AUTHORITY, required_consecutive_cycles=self.hz['required_consecutive_cycles'],
            extra={'test_only_stub': self.stub, 's56_search': {'search_id': self.spec['search_id'],
                                                               'launch_index': launch_index, 'depth': depth,
                                                               'action': action}})
        by_label = {e['label']: e for e in hspec['candidates']}
        for lab, z in zip(labels, action['zs']):
            e = by_label[lab]
            if e['eval_key'] != self.eval_key(z, depth) or e['key'] != self.lattice.key(tuple(z)):
                raise Refused(f'launch {launch_index}: the harness-frozen key of {lab} does not recompute '
                              f'(eval {e["eval_key"][:16]}, candidate {e["key"][:16]})')
        self.lock_content = set_lock_spec(self.lock_path, self.lock_content, sha)
        ctx = self._ctx(launch_dir, spec_path, sha, hspec, launch_index)
        hooks.during_evaluate_start()
        H.evaluate(labels, ctx, heartbeat_tag=f'launch_{launch_index:03d}')
        return [{'z': list(z), 'depth': depth, 'eval_key': by_label[lab]['eval_key'],
                 'eval_dir': os.path.join(launch_dir, 'evals', by_label[lab]['eval_dir'])}
                for lab, z in zip(labels, action['zs'])]

    def verify_launch(self, launch_dir, action):
        """Replay: the launch's frozen harness spec holds exactly the action's entries, with the recomputed keys."""
        specs = [n for n in os.listdir(launch_dir) if n.startswith('campaign_spec_') and n.endswith('.json')]
        if len(specs) != 1:
            return [f'{len(specs)} harness specs in the launch dir']
        hspec = load_json(os.path.join(launch_dir, specs[0]))
        got = sorted((e['key'], e['eval_key']) for e in hspec['candidates'])
        want = sorted((self.lattice.key(tuple(z)), self.eval_key(z, action['depth'])) for z in action['zs'])
        out = [] if got == want else ['the harness spec entries are not the action recomputed']
        if (hspec.get('extra') or {}).get('test_only_stub') is not self.stub:
            out.append('the harness spec stub flag differs from the oracle')
        return out

    def summary_from_dir(self, eval_dir, z, depth):
        rec_p = os.path.join(eval_dir, 'evaluation_record.json')
        bar_p = os.path.join(eval_dir, 'parent_barrier_record.json')
        src = rec_p if os.path.isfile(rec_p) else (bar_p if os.path.isfile(bar_p) else None)
        if src is None:
            raise Refused(f'{eval_dir}: no evaluation record')
        rec = load_json(src)
        with open(os.path.join(eval_dir, 'exit_code.txt')) as handle:
            exit_code = int(handle.read().strip())
        if rec.get('candidate_key') != self.lattice.key(tuple(z)):
            raise Refused(f'{eval_dir}: record candidate key {str(rec.get("candidate_key"))[:16]} is not z\'s')
        if not os.path.basename(eval_dir).startswith(self.eval_key(z, depth)[:16]):
            raise Refused(f'{eval_dir}: eval dir does not carry the recomputed eval key')
        if self.stub:
            if rec.get('status') == 'stub' and rec.get('stub') is True and exit_code == 0:
                return dict(self.land.summary(z, depth), stub_record=True), src
            err = {'depth': depth, 'status': 'harness_error', 'F': None, 'bar': None, 'stub_record': True,
                   'harness_status': rec.get('status'), 'exit_code': exit_code}
            return err, src
        if src != rec_p or rec.get('eval_key') != self.eval_key(z, depth):
            return {'depth': depth, 'status': 'harness_error', 'F': None, 'bar': None,
                    'reason': 'no evaluation record (barrier only) or record eval key != recomputed'}, src
        return map_record(read_eval_files(eval_dir), depth, self.mapping), src


def make_oracle(spec, lattice, lock_path=None):
    kind = spec['oracle']['kind']
    if kind == 'synthetic':
        return SyntheticOracle(spec, lattice)
    if kind in ('stub_children', 'harness'):
        return HarnessOracle(spec, lattice, lock_path)
    raise Refused(f'unknown oracle {kind!r}')


# ======================================================================================================================
#  THE ORACLE MAPPING: harness records -> the policy's summaries (S56_SEARCH_DRIVER_DESIGN.md, "Oracle mapping and key
#  rules", Planner decisions 2026-11-23, acc80a2e). Pure functions of the files in one eval dir; no model import.
# ======================================================================================================================
#  F        F(k) = I(x) + recourse(k) = I(x) + gross(k) - TV(k), EUR of total discounted planning cost: `recourse` and
#           `gross_operational_cost` / `terminal_salvage_value` are production's per-cycle trajectory fields
#           (shared_resources_planning.py:3386-3388, row 3588-3590: _get_operational_recourse_components), I(x) the
#           entry's `s56_investment.investment_cost_eur` (p516_lattice.GeneralLattice.investment_cost). The per-cycle TV
#           credit is therefore captured already; no new capture. Checked: recourse(k) == gross(k) - TV(k) exactly, and
#           F(terminal) == the cost breakdown's total_planning_cost_eur (schema v1 or v2 convention, stated).
#  bar_F    max |F(k) - F(k-1)| over the BAR_WINDOW cycles ending at the read cycle (the predecessor of the first window
#           row from the full trajectory); None when a step is unavailable. The triangle bound bar_gross + range(TV) is
#           reported beside it (bar_F <= the bound is checked), never used.
#  exit     read at the record's certification cycle (an exit-depth run) or at the WOULD-BE certification cycle of a
#           stop-disabled run: the first cycle with consecutive_converged_cycles >= required_consecutive_cycles.
#  settled  run to the cap K with `s56_settled_stop_disabled`: SETTLED iff (i) cycles_run == K, (ii) each of the last W
#           cycles ran at tail depth (convergence_depth_tail_state.json per_cycle[k].active), (iii) production's
#           per-cycle certification test held at each of them (cycle_convergence = boyd_all_pass and local_solves_ok,
#           shared_resources_planning.py:3469 -- the test that advances the certificate counter; W = 10 = the
#           certificate length, so (iii) is 'the run would have certified at K'), (iv) guards exact (the child's solve
#           profile reconciles -- `solve_profile_reconciliation`, mapping v2 --, nothing blocked, the stop-disabling
#           hook summary ok, the tail state matches its
#           declaration) and no local-solve failure. Otherwise 'not_settled' (the design's "uncertified").
#  S        max - min of F over the last W cycles ending at K.
#  PR       signed priced consensus residual, Expert ruling 3 sign: the first-order estimate is Q - price + dual, so
#           PR = -price + dual with
#             price = -sum_b sum_agents w_b x settlement_total (production _get_operational_interface_settlement_blocks
#                     'total'; the settlement price part Q's convention EXCLUDES -- see SETTLEMENT_DERIVATION),
#             dual  = sum over interface blocks / periods of the TSO-side terms
#                       sigma [lam_p dP / (S_b,T R_MVA) + lam_q dQ / (S_b,T R_MVA) + lam_v dV / V_b^2],
#                     dP = P_tso - P_dso etc. (the step 4 evaluator's expectation-channel form,
#                     p515_s56_step4_main_evaluate.priced_residual), plus the three ESS channels
#                       sigma sum_{i in tso, dso, esso} lam_i (x_i - z) / (2 S_i)   for P and Q,
#                     each the agent's own augmented-Lagrangian dual term at the terminal consensus (production
#                     _update_shared_energy_storage_variables, normalisation a_i = 1 / (2 S_i)).
#           sigma = the common objective scale: one scaled-objective unit of any agent = sigma EUR.
#  A        sum_b |price_b| (per (year, day) block) + the sum of |each dual term|.
#  G        ESSO P and Q slacks priced at the ESS multiplier: sigma |lam_esso| |s_up - s_down| / (2 S_esso);
#           + the within-cohort leak priced as the conversion loss (cost breakdown leak_detector.priced_loss_eur);
#           + the network closure slack priced at its row dual: sigma |dual| |s_up - s_down| (p.u.).
#  gates    status ok and guards exact; max |P_tso - P_dso| <= 12.5 kW; ESSO slack gate max |s_up - s_down| <=
#           ESSO_SLACK_TOL; closure gate max |s_up - s_down| <= TAU_CL_PU (p.u. of the block base, i.e. MWh <= TAU_CL_PU x
#           baseMVA); the D1 efficiency-equality gate (one (eta_ch, eta_dch) per node over its active cohorts).
#  leak     lambda L (conversion-loss price) <= 25 EUR: REPORTED and flagged, not in gates_pass (TASKS 28ba36b4).
SETTLEMENT_DERIVATION = (
    'The EV record\'s Q is production gross_operational_cost = gross_operational_cost_including_settlement - '
    'interface_settlement_contracted_total - voltage_pin_total (shared_resources_planning.py:1305-1307, in '
    '_get_operational_recourse_components); the settlement-excluded-in-full form replaces it only when '
    '_settlement_excluded_in_full_convention returns a value (:1316-1320), i.e. only when the per-scenario coupling '
    'is wired (:4899-4900, _per_scenario_interface_coupling_wired :4882), which it never is on the one-scenario EV '
    'twin (_per_scenario_interface_coupling_active :4406; EV harness report, Blocked 3). At one market scenario the '
    'deviation part is identically zero and the contracted part IS the whole settlement (shared_resources_planning.py:'
    '1274-1275, the Addendum 38 (C) comment; _get_local_interface_settlement :826-827). So the EV Q contains NO '
    'settlement price part: it excludes the whole settlement, exactly as the coupled step 4 arm\'s Q did, and the '
    'price term is the full weighted settlement, price = -sum w_b settlement_total (= -interface_settlement_total of '
    'the record\'s recourse_components). Checked per record: |deviation total| == 0 and contracted total == total.')
# v2 (TASKS 5b, 2026-11-25): guards['solve_profile_identity_holds'] reads `solve_profile_reconciliation` (the ESSO
# recovery solves counted exactly); the summaries carry its detail. v1: the harness's identity_holds alone.
MAPPING_SCHEMA = 'p515_s56_oracle_mapping_v2'
MAPPING_PARAMETERS = {
    'W': 10, 'bar_window': 10, 'interface_p_ceiling_mw': 0.0125, 'leak_ceiling_eur': 25.0,
    # FINAL (integration ruling 1, S56_SEARCH_DRIVER_DESIGN.md "Integration rulings", 601775c0: "confirmed as built";
    # both are backstops, the priced slacks enter G in any case): tau_cl = 1e-5 p.u. of the block base (1e-3 MWh on
    # base 100); ESSO slack gate |s_up - s_down| <= 1e-5 MW / MVAr per element. Both equal production's own equality
    # tolerance definitions.EQUALITY_TOLERANCE = 1e-5 (the band production imposes on the closure when its slack is
    # not built, model_construction_helpers.sess_soc_final_rule).
    'tau_cl_pu': 1e-5, 'esso_slack_tol': 1e-5,
    'provisional': [],
    'authority': 'S56_SEARCH_DRIVER_DESIGN.md, Integration rulings 1 and 4 (601775c0)',
}
CORRECTED_COHORT_MODES = ('independent_units', 'independent_units_all_node_years')
SETTLED_STATUS, NOT_SETTLED_STATUS = 'settled', 'not_settled'
# The identity tolerance (Expert-adopted, P5_15_S56_STEP4_EXPERT_HANDBACK_3.md ruling 2; applied to the settlement
# checks by the Planner's ruling of the S56 pilot evaluation): |a - b| <= max(c u sum|summands|, floor), c = 64,
# u = 2^-53, floor 1e-6 EUR; the summands are the per-block settlement totals. NOT a mapping parameter (a frozen spec's
# `mapping` does not carry it): it bounds float64 summation-order rounding only. Why it is needed: the per-block
# totals cancel (TSO ~ -1.3e9 EUR against DSO ~ +1.3e9 EUR to a net of 6-72 EUR on the pilot), production sums them
# TSO-first then per DSO node (shared_resources_planning.py:1254-1260) while the mapping sums the captured blocks in
# list order, and the two orders differ by ~1-2 ulp of 1.3e9 (2.4e-7 EUR); the old 1e-9 x max(1, |total|) form gave
# 6e-9 EUR on a 6 EUR net. Both sides are float64 (the production-order sum of the captured blocks reproduces the
# record's interface_settlement_total bitwise on all 7 pilot records).
IDENTITY_TOL_C, IDENTITY_TOL_U, IDENTITY_TOL_FLOOR_EUR = 64, 2.0 ** -53, 1e-6


def identity_tolerance(summands):
    """max(64 u sum|summands|, 1e-6 EUR) -- the identity tolerance above."""
    return max(IDENTITY_TOL_C * IDENTITY_TOL_U * sum(abs(v) for v in summands), IDENTITY_TOL_FLOOR_EUR)


def _jsonl(path):
    with open(path) as handle:
        return [json.loads(t) for t in handle if t.strip()]


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float('inf'),
                                                                                            float('-inf'))


def read_eval_files(eval_dir):
    """The files of one real eval dir the mapping reads (missing -> None)."""
    out = {'eval_dir': eval_dir}
    for name, key, reader in (('evaluation_record.json', 'record', load_json),
                              ('per_cycle_record.jsonl', 'rows', _jsonl),
                              ('convergence_depth_tail_state.json', 'tail', load_json),
                              ('s56_cost_breakdown.json', 'breakdown', load_json),
                              ('s56_pricing_terminal.json', 'pricing', load_json)):
        p = os.path.join(eval_dir, name)
        out[key] = reader(p) if os.path.isfile(p) else None
    p = os.path.join(eval_dir, 'exit_code.txt')
    out['exit_code'] = int(open(p).read().strip()) if os.path.isfile(p) else None
    # the ESSO recovery events of the run (read by `solve_profile_reconciliation`): the sibling of the record's network
    # failure file, named by the harness's own rule (p515_g_g1_g4_admm_gates._scan_and_write_network_failures:
    # 'network_failures_' -> 'esso_recovery_events_'), looked up inside THIS eval dir. None when the record names no
    # failure file or the file is absent (a missing file is never read as "no event").
    nf = ((out['record'] or {}).get('network_failures_summary') or {}).get('path')
    name = os.path.basename(nf).replace('network_failures_', 'esso_recovery_events_', 1) if isinstance(nf, str) \
        else None
    p = os.path.join(eval_dir, name) if name and name.startswith('esso_recovery_events_') else None
    out['esso_recovery_events_file'] = name
    out['esso_recovery_events'] = _jsonl(p) if p and os.path.isfile(p) else None
    return out


def f_series(record, rows):
    """{cycle: F} and the per-cycle checks (module comment)."""
    inv = ((record or {}).get('s56_investment') or {}).get('investment_cost_eur')
    if not _fin(inv):
        raise Refused('the record carries no s56_investment.investment_cost_eur (I(x))')
    out, bad = {}, []
    for r in rows:
        k, net, gross, tv = r.get('cycle'), r.get('recourse'), r.get('gross_operational_cost'), \
            r.get('terminal_salvage_value')
        if net is None:
            out[k] = None
            continue
        if gross - tv != net:
            bad.append(k)
        out[k] = inv + net
    return inv, out, bad


def bar_at(F, c, window):
    """max |F(k) - F(k-1)| for k in the `window` cycles ending at c (None if any step is unavailable)."""
    steps = []
    for k in range(max(1, c - window + 1), c + 1):
        a, b = F.get(k), F.get(k - 1)
        if a is None or b is None:
            if k == 1:
                continue     # cycle 1 has no predecessor (the harness bar's rule)
            return None, steps
        steps.append(abs(a - b))
    return (max(steps) if steps else None), steps


def would_be_certification_cycle(rows, required):
    for r in rows:
        if isinstance(r.get('consecutive_converged_cycles'), int) and r['consecutive_converged_cycles'] >= required:
            return r['cycle']
    return None


def triangle_bound(rows, c, window):
    by = {r['cycle']: r for r in rows}
    gsteps, tvs = [], []
    for k in range(max(1, c - window + 1), c + 1):
        if k in by and by[k].get('terminal_salvage_value') is not None:
            tvs.append(by[k]['terminal_salvage_value'])
        if k - 1 in by and k in by and by[k].get('gross_operational_cost') is not None and \
                by[k - 1].get('gross_operational_cost') is not None:
            gsteps.append(abs(by[k]['gross_operational_cost'] - by[k - 1]['gross_operational_cost']))
    if not gsteps or not tvs:
        return None
    return max(gsteps) + (max(tvs) - min(tvs))


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


# SOLVE-PROFILE RECONCILIATION (Planner decision, TASKS 5b, 2026-11-25: "extend the reconciliation exactly (ESSO
# recovery solves counted; every failure recovered) -- harness accounting, not formulation"). The harness's identity
# (p515_g_g1_g4_admm_gates.py, "W35 item 3": observed == base + retry_solves_credited, retries = the network-failure
# events' [recovery_attempted] + [tier2_attempted]) is declared UNSUPPORTED (identity_holds False) whenever an ESSO
# recovery event exists, because "ESSO recovery events carry no attempt count". They do carry it, in the event record
# the production ESSO path writes (shared_energy_storage_data._optimize): an event is appended to the diagnostic sink
# iff `recovery_attempted` (the primary IPOPT solve ended internalSolverError / maxIterations / infeasible), and then
# exactly one tier-1 cold retry ran (`_run_solver_attempt(..., log_suffix='recovery')`, one OptSolver.solve) plus one
# tier-2 retry iff `tier2_attempted` (`log_suffix='recovery_tier2'`); the guard counts every OptSolver.solve entry
# (p513_solve_profile_guard.guarded_solve), raising or not. So one ESSO event costs 1 + [tier2_attempted] solves, and
# `recovery_succeeded` is solver_result_succeeded of the solution finally loaded. The harness writes the events to
# esso_recovery_events_<label>.jsonl (every sink entry, tagged with its round) and counts them in
# network_failures_summary.n_esso_recovery_events. The guard holds iff the harness identity holds (unchanged), or -- only
# when the harness declared the reconciliation unsupported BECAUSE of >= 1 ESSO event -- ALL of:
#   (1) the extended identity, exactly: observed == base_solves + retry_solves_credited + sum_events (1 + [tier2]);
#   (2) every network failure recovered: classes recovered_tier1 + recovered_tier2 == n_blocks, no unrecovered /
#       not_attempted / indeterminate block, and retry_solves_credited == recovered_tier1 + 2 x recovered_tier2 (the
#       harness's per-event credit of an all-recovered set);
#   (3) every ESSO event recorded as succeeded (`recovery_succeeded` is True), with a consistent tier record, and the
#       events file holds exactly n_esso_recovery_events events.
# Otherwise the guard fails, as before. Computed here from fields the harness already records; the harness is unchanged.
SOLVE_PROFILE_RECONCILIATION_RULE = (
    'harness identity (observed == base + network retries credited) holds; or, when it is unsupported because of >= 1 '
    'ESSO recovery event: observed == base + retry_solves_credited + sum over ESSO events of (1 + [tier2_attempted]) '
    'exactly, every network failure recovered (recovered_tier1 + recovered_tier2 == n_blocks; credited == tier1 + 2 '
    'tier2), every ESSO event recovery_succeeded True, events file count == n_esso_recovery_events')


def solve_profile_reconciliation(record, files):
    """The solve-count reconciliation of one record (section comment above). Pure. -> {'holds', 'via', ...}."""
    sp = (record or {}).get('solve_profile') or {}
    if sp.get('identity_holds') is True:
        return {'holds': True, 'via': 'harness_identity', 'failed': []}
    nfs = (record or {}).get('network_failures_summary') or {}
    cls = nfs.get('classes') or {}
    obs = (sp.get('observed') or {}).get('permitted_solve')
    base, cred, n_esso = sp.get('base_solves'), sp.get('retry_solves_credited'), nfs.get('n_esso_recovery_events')
    events = files.get('esso_recovery_events')
    failed = []
    if not all(_int(v) for v in (obs, base, cred, n_esso)):
        failed.append('observed / base_solves / retry_solves_credited / n_esso_recovery_events not all integers')
    if not (_int(n_esso) and n_esso >= 1):
        failed.append('no ESSO recovery event: the harness identity is the rule, and it does not hold')
    if sp.get('reconciliation_supported') is not False:
        failed.append('the harness did not declare the reconciliation unsupported')
    t1, t2 = cls.get('recovered_tier1'), cls.get('recovered_tier2')
    n_blocks = nfs.get('n_blocks')
    net_ok = (all(_int(v) for v in (t1, t2, n_blocks)) and t1 + t2 == n_blocks
              and all(cls.get(k) == 0 for k in ('unrecovered', 'not_attempted', 'indeterminate')))
    if not net_ok:
        failed.append('a network failure is not recovered (or the classes are incomplete)')
    elif not (_int(cred) and cred == t1 + 2 * t2):
        failed.append('retry_solves_credited != recovered_tier1 + 2 x recovered_tier2')
    esso_solves, per_event = None, []
    if events is None:
        failed.append(f"ESSO recovery events file {files.get('esso_recovery_events_file')!r} not found")
    else:
        if not (_int(n_esso) and len(events) == n_esso):
            failed.append(f'events file holds {len(events)} events, the summary counts {n_esso}')
        esso_solves = 0
        for ev in events:
            t2a = ev.get('tier2_attempted')
            ok = (ev.get('record_type') == 'esso_recovery' and ev.get('subsystem') == 'esso'
                  and isinstance(t2a, bool) and ev.get('tier') == ('tier2' if t2a else 'tier1')
                  and ev.get('recovery_succeeded') is True)
            per_event.append({'node_id': ev.get('node_id'), 'cycle': ev.get('cycle'), 'tier': ev.get('tier'),
                              'tier2_attempted': t2a, 'recovery_succeeded': ev.get('recovery_succeeded'),
                              'extra_solves': 1 + int(t2a is True), 'ok': ok})
            esso_solves += 1 + int(t2a is True)
            if not ok:
                failed.append(f"ESSO event node {ev.get('node_id')} cycle {ev.get('cycle')}: not recorded as a "
                              f"succeeded recovery with a consistent tier")
    expected = (base + cred + esso_solves) if (_int(base) and _int(cred) and esso_solves is not None) else None
    identity = expected is not None and _int(obs) and obs == expected
    if not identity:
        failed.append(f'observed {obs} != base {base} + credited {cred} + ESSO recovery solves {esso_solves}')
    return {'holds': not failed, 'via': 'extended_esso_recovery', 'failed': failed, 'observed': obs,
            'base_solves': base, 'retry_solves_credited': cred, 'esso_recovery_solves': esso_solves,
            'expected_solves': expected, 'n_esso_recovery_events': n_esso, 'esso_events': per_event,
            'network_classes': {k: cls.get(k) for k in ('recovered_tier1', 'recovered_tier2', 'unrecovered',
                                                        'not_attempted', 'indeterminate')}, 'n_blocks': n_blocks,
            'rule': SOLVE_PROFILE_RECONCILIATION_RULE}


def guards_exact(record, files):
    sp = (record or {}).get('solve_profile') or {}
    obs = sp.get('observed') or {}
    checks = {
        'child_exit_0': files.get('exit_code') == 0,
        'solve_profile_identity_holds': solve_profile_reconciliation(record, files)['holds'],
        'nothing_blocked': obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0,
        'no_local_solve_failure': (record or {}).get('local_solve_failures') == 0,
        'tail_state_matches_declaration': ((record or {}).get('convergence_depth_tail_state_check') or {}).get('match')
                                          is True,
    }
    if (record or {}).get('s56_settled_stop_disabled') is True:
        checks['stop_disabling_summary_ok'] = ((record or {}).get('s56_certification_stop_summary') or {}).get('ok') \
            is True
    return checks


def pricing_components(pricing, breakdown, record, params=MAPPING_PARAMETERS):
    """PR, A, G and the pricing-based gates from the terminal capture (module comment). Pure."""
    sigma = pricing['sigma']
    dual_terms, absf = 0.0, 0.0
    parts = {'interface_p': 0.0, 'interface_q': 0.0, 'interface_v': 0.0, 'ess_p': 0.0, 'ess_q': 0.0}
    max_dp = max_dq = max_dv = 0.0
    for b in pricing['interface']:
        fac_pq = sigma / (b['tso_base_mva'] * b['rating_mva'])
        fac_v = sigma / (b['v_base_kv'] ** 2)
        for t in range(len(b['p_tso'])):
            dP, dQ = b['p_tso'][t] - b['p_dso'][t], b['q_tso'][t] - b['q_dso'][t]
            dV = b['v_tso'][t] - b['v_dso'][t]
            max_dp, max_dq, max_dv = max(max_dp, abs(dP)), max(max_dq, abs(dQ)), max(max_dv, abs(dV))
            for name, term in (('interface_p', b['lambda_p_tso'][t] * dP * fac_pq),
                               ('interface_q', b['lambda_q_tso'][t] * dQ * fac_pq),
                               ('interface_v', b['lambda_v_tso'][t] * dV * fac_v)):
                parts[name] += term
                dual_terms += term
                absf += abs(term)
    for b in pricing['ess']:
        for k in ('p', 'q'):
            z = b[f'z_{k}']
            for agent in ('tso', 'dso', 'esso'):
                x, lam, s_i = b[f'x_{k}_{agent}'], b[f'lambda_{k}_{agent}'], b['rating_mva'][agent]
                for t in range(len(z)):
                    term = sigma * lam[t] * (x[t] - z[t]) / (2.0 * s_i)
                    parts[f'ess_{k}'] += term
                    dual_terms += term
                    absf += abs(term)
    price_b = {}
    dev_total = contracted_total = total = 0.0
    for s in pricing['settlement_blocks']:
        bk = f"{s['year']}|{s['day']}"
        price_b[bk] = price_b.get(bk, 0.0) - s['total']
        total += s['total']
        contracted_total += s['contracted']
        dev_total += s['deviation']
    price = sum(price_b.values())
    pr = -price + dual_terms
    a = sum(abs(v) for v in price_b.values()) + absf
    # G
    ess_rating = {(b['node_id'], b['year'], b['day']): b for b in pricing['ess']}
    g_slack, max_slack = 0.0, 0.0
    for s in pricing['esso_slacks']:
        eb = ess_rating.get((s['node_id'], s['year'], s['day']))
        for k in ('p', 'q'):
            if s.get(f'{k}_up') is None:
                continue
            for t, (u, d) in enumerate(zip(s[f'{k}_up'], s[f'{k}_down'])):
                net = abs(u - d)
                max_slack = max(max_slack, net)
                if eb is None:
                    raise Refused(f"ESSO slack block {s['node_id']} {s['year']} {s['day']} has no ESS consensus block")
                g_slack += sigma * abs(eb[f'lambda_{k}_esso'][t]) * net / (2.0 * eb['rating_mva']['esso'])
    g_closure, max_cl_pu, max_cl_mwh, no_dual, inactive, no_slack, zero_no_dual = 0.0, 0.0, 0.0, 0, 0, 0, 0
    for c in pricing['closure']:
        if c.get('active') is False:      # a deactivated row constrains nothing: counted, not priced, not gated
            inactive += 1
            continue
        if c['slack_up_pu'] is None or c['slack_down_pu'] is None:
            no_slack += 1                 # the row is built without the slack pair (no relaxation exists)
            continue
        net = abs(c['slack_up_pu'] - c['slack_down_pu'])
        max_cl_pu, max_cl_mwh = max(max_cl_pu, net), max(max_cl_mwh, net * c['base_mva'])
        if c['row_dual'] is None:
            if net == 0.0:                # no relaxation to price: exactly 0, counted
                zero_no_dual += 1
            else:
                no_dual += 1              # a relaxation without its price: the summary is incomplete
            continue
        g_closure += sigma * abs(c['row_dual']) * net
    leak = (breakdown or {}).get('leak_detector') or {}
    g_leak = leak.get('priced_loss_eur')
    if not _fin(g_leak):
        raise Refused('the cost breakdown carries no leak_detector.priced_loss_eur')
    rc = (record or {}).get('recourse_components') or {}
    # Both identity checks use the identity tolerance (module constant IDENTITY_TOL_*; summands = the per-block
    # settlement totals). Before the pilot evaluation ruling both used 1e-9 x max(1, |total|).
    tol = identity_tolerance(s['total'] for s in pricing['settlement_blocks'])
    rec_total = rc.get('interface_settlement_total')
    settle = {'interface_settlement_total_record': rec_total,
              'sum_blocks_total': total, 'contracted_total': contracted_total, 'deviation_total': dev_total,
              'deviation_is_zero': dev_total == 0.0,
              'contracted_equals_total': abs(contracted_total - total) <= tol,
              'blocks_reconcile_with_record': (rec_total is not None and abs(total - rec_total) <= tol),
              'identity_tolerance_eur': tol,
              'abs_diff_contracted_minus_total_eur': abs(contracted_total - total),
              'abs_diff_blocks_minus_record_eur': None if rec_total is None else abs(total - rec_total),
              'identity_tolerance_rule': ('max(64 u sum|per-block settlement totals|, 1e-6 EUR), u = 2^-53 '
                                          '(Expert handback 3 ruling 2)')}
    return {
        'PR': pr, 'A': a, 'G': g_slack + g_leak + g_closure,
        'PR_parts': {'minus_price': -price, 'dual': dual_terms, **parts},
        'A_parts': {'abs_price_blocks': sum(abs(v) for v in price_b.values()), 'abs_dual_terms': absf},
        'G_parts': {'esso_slack': g_slack, 'leak_loss': g_leak, 'closure': g_closure,
                    'closure_rows_without_dual': no_dual, 'closure_rows_inactive': inactive,
                    'closure_rows_without_slack': no_slack, 'closure_rows_zero_slack_without_dual': zero_no_dual,
                    'closure_rows': len(pricing['closure'])},
        'settlement_check': settle, 'sigma': sigma,
        'interface_max_abs': {'dP_mw': max_dp, 'dQ_mvar': max_dq, 'dV_kv': max_dv},
        'esso_slack_max_abs_net': max_slack, 'closure_max_abs_net_pu': max_cl_pu, 'closure_max_abs_net_mwh': max_cl_mwh,
        'leak': {'priced_loss_eur': g_leak, 'priced_full_eur': leak.get('priced_eur'),
                 'L_weighted_mwh': leak.get('L_weighted_mwh'), 'ceiling_eur': params['leak_ceiling_eur'],
                 'within_ceiling': g_leak <= params['leak_ceiling_eur'], 'gate': False,
                 'note': 'reported and flagged; not in gates_pass until the pilot measures it (TASKS 28ba36b4)'},
    }


def d1_gate(record, pricing):
    plan = record.get('s56_cohort_formulation')
    active = (record.get('s56_investment') or {}).get('active_cohorts') or []
    per = {}
    for n, y in active:
        eff = ((pricing or {}).get('efficiency') or {}).get(str(n), {}).get(str(y))
        per.setdefault(str(n), set()).add(tuple(eff) if eff else None)
    ok = all(None not in s and len(s) <= 1 for s in per.values())
    return {'cohort_formulation': plan, 'applies': plan in CORRECTED_COHORT_MODES,
            'per_node_pairs': {k: sorted(map(list, (x for x in v if x))) for k, v in per.items()},
            'pass': ok if plan in CORRECTED_COHORT_MODES else True}


def map_record(files, depth, params=MAPPING_PARAMETERS, read_cycle=None):
    """THE MAPPING of one real eval dir (`read_eval_files`) at `depth` -> the policy summary (module comment).
    exit: {'depth', 'status' certified | not_certified | harness_error, 'F', 'bar', ...}; settled: {'depth', 'status'
    settled | not_settled | harness_error, 'F', 'V': {S, PR, G, A}, 'gates_pass', 'gates', 'settled_checks', ...}.
    A missing file or field is 'harness_error' with the reason (never a silent default)."""
    rec, rows = files.get('record'), files.get('rows')
    base = {'depth': depth, 'mapping_schema': MAPPING_SCHEMA,
            'objective_convention': ('F = I(x) + net = I(x) + gross_operational_cost - terminal value credit, EUR of '
                                     'total discounted planning cost; gross settlement-excluded (contracted part and '
                                     'pin excluded; at one scenario the whole settlement)')}
    missing = [k for k in ('record', 'rows') if not files.get(k)]
    if depth == P.DEPTH_SETTLED:
        missing += [k for k in ('tail', 'breakdown', 'pricing') if not files.get(k)]
    if files.get('exit_code') is None:
        missing.append('exit_code')
    if missing or files.get('exit_code') != 0 or (rec or {}).get('status') == 'error':
        return dict(base, status='harness_error', F=None, bar=None,
                    reason=f"missing {missing}" if missing else f"exit {files.get('exit_code')} status "
                                                                f"{(rec or {}).get('status')}")
    try:
        inv, F, bad = f_series(rec, rows)
    except Refused as error:
        return dict(base, status='harness_error', F=None, bar=None, reason=str(error))
    cycles = sorted(F)
    required = int((rec.get('certification') or {}).get('required_consecutive_cycles') or
                   rec.get('required_consecutive_cycles'))
    guards = guards_exact(rec, files)
    recon = solve_profile_reconciliation(rec, files)   # the detail behind guards['solve_profile_identity_holds']
    stop_disabled = rec.get('s56_settled_stop_disabled') is True
    if depth == P.DEPTH_EXIT:
        if read_cycle is None:
            read_cycle = (would_be_certification_cycle(rows, required) if stop_disabled
                          else (rec.get('certification') or {}).get('certification_cycle'))
        status = 'certified' if (read_cycle is not None and all(guards.values())
                                 and (stop_disabled or rec.get('status') == 'certified')) else 'not_certified'
        c = read_cycle if read_cycle is not None else (cycles[-1] if cycles else None)
        Fc = F.get(c) if c is not None else None
        bar, steps = bar_at(F, c, params['bar_window']) if c is not None else (None, [])
        tri = triangle_bound(rows, c, params['bar_window']) if c is not None else None
        return dict(base, status=status, F=Fc, bar=bar, read_cycle=c,
                    read_at=('terminal cycle (no certification cycle: uncertified)' if read_cycle is None
                             else 'would-be certification cycle (stop disabled)' if stop_disabled
                             else 'certification cycle'),
                    I=inv, bar_steps_n=len(steps), bar_triangle_bound=tri,
                    bar_within_triangle_bound=(tri is None or bar is None or bar <= tri * (1 + 1e-12) + 1e-9),
                    recourse_identity_bad_cycles=bad, guards=guards, solve_profile_reconciliation=recon,
                    cycles_run=rec.get('cycles_run'), wall_s=(rec.get('wall_time_s') or {}).get('child_process_s'))
    # settled
    K = rec.get('cap')
    W = params['W']
    last = list(range(K - W + 1, K + 1)) if isinstance(K, int) else []
    by = {r['cycle']: r for r in rows}
    tail_active = {p.get('cycle'): bool(p.get('active')) for p in (files['tail'].get('per_cycle') or [])}
    checks = {
        'reached_K': rec.get('cycles_run') == K and len(cycles) == K,
        'last_W_at_tail_depth': bool(last) and all(tail_active.get(k) is True for k in last),
        'would_have_certified_each_of_last_W': bool(last) and all(
            (by.get(k) or {}).get('cycle_convergence') is True for k in last),
        'guards_exact_and_no_local_failure': all(guards.values()),
        'stop_disabled_declared': stop_disabled,
    }
    checks['K_at_least_W'] = isinstance(K, int) and K >= W
    # S over the last W cycles (over every cycle when K < W: such a run is never settled; S_window_cycles says so)
    s_win = list(range(max(1, K - W + 1), K + 1)) if isinstance(K, int) else []
    Fw = [F.get(k) for k in s_win]
    S = (max(Fw) - min(Fw)) if Fw and all(v is not None for v in Fw) else None
    try:
        pc = pricing_components(files['pricing'], files['breakdown'], rec, params)
    except (Refused, KeyError, TypeError, ZeroDivisionError) as error:
        return dict(base, status='harness_error', F=None, bar=None, reason=f'pricing: {type(error).__name__}: {error}')
    d1 = d1_gate(rec, files['pricing'])
    gates = {
        'status_ok_and_guards_exact': all(guards.values()) and rec.get('status') in ('certified', 'not_certified'),
        'interface_p_ceiling_12p5kw': pc['interface_max_abs']['dP_mw'] <= params['interface_p_ceiling_mw'],
        'esso_slack_gate': pc['esso_slack_max_abs_net'] <= params['esso_slack_tol'],
        'network_closure_slack_le_tau_cl': pc['closure_max_abs_net_pu'] <= params['tau_cl_pu'],
        'd1_efficiency_equality': d1['pass'],
    }
    complete = S is not None and all(_fin(v) for v in (pc['PR'], pc['A'], pc['G'])) and \
        pc['G_parts']['closure_rows_without_dual'] == 0 and pc['settlement_check']['deviation_is_zero'] and \
        pc['settlement_check']['contracted_equals_total'] and pc['settlement_check']['blocks_reconcile_with_record']
    if not complete:
        return dict(base, status='harness_error', F=F.get(K), bar=None, V=None, gates_pass=None,
                    reason='incomplete settled summary', S=S, pricing=pc, settled_checks=checks)
    exit_view = map_record(files, P.DEPTH_EXIT, params)
    status = SETTLED_STATUS if all(checks.values()) else NOT_SETTLED_STATUS
    bk = files['breakdown']
    total_bd = bk.get('total_planning_cost_eur')
    return dict(base, status=status, F=F.get(K), K=K, V={'S': S, 'PR': pc['PR'], 'G': pc['G'], 'A': pc['A']},
                gates_pass=all(gates.values()), gates=gates, settled_checks=checks,
                not_settled_reasons=sorted(k for k, v in checks.items() if not v), S_window_cycles=len(s_win),
                V_parts={'PR': pc['PR_parts'], 'A': pc['A_parts'], 'G': pc['G_parts']},
                leak=pc['leak'], leak_flag=not pc['leak']['within_ceiling'],
                interface_max_abs=pc['interface_max_abs'], esso_slack_max_abs_net=pc['esso_slack_max_abs_net'],
                closure_max_abs_net_mwh=pc['closure_max_abs_net_mwh'], d1=d1, settlement_check=pc['settlement_check'],
                sigma=pc['sigma'], I=inv,
                F_equals_breakdown_total=(_fin(total_bd) and F.get(K) is not None
                                          and abs(total_bd - F[K]) <= 1e-9 * abs(total_bd)),
                breakdown_schema=bk.get('schema'), recourse_identity_bad_cycles=bad,
                exit_view={k: exit_view.get(k) for k in ('status', 'F', 'bar', 'read_cycle', 'read_at')},
                mapping_parameters=params, guards=guards, solve_profile_reconciliation=recon,
                cycles_run=rec.get('cycles_run'), wall_s=(rec.get('wall_time_s') or {}).get('child_process_s'))


def _code_text(name):
    p = os.path.join(CODE_ROOT, name)
    return open(p).read() if os.path.isfile(p) else ''


def assert_mapping_capture_paths(H, hz=None):
    """Rule eleven for the mapping, BEFORE any real launch: every input the mapping reads has a capture path in the
    harness as committed, and the search's settled depth declares the keys that produce them. -> checklist; raises
    Refused naming the failed checks."""
    import inspect
    rec_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    checks = {
        'trajectory_fields': {'cycle', 'recourse', 'gross_operational_cost', 'terminal_salvage_value',
                              'cycle_convergence', 'consecutive_converged_cycles', 'local_solves_ok'}
                             <= set(H.PER_CYCLE_TRAJECTORY_FIELDS),
        'record_fields': all(f"'{k}'" in rec_src for k in ('solve_profile', 'local_solve_failures', 'certification',
                                                            'cycles_run', 'recourse_components', 'cap', 'status')),
        'tail_state_file_written': 'CONVERGENCE_DEPTH_TAIL_STATE_FILE' in inspect.getsource(
            H.persist_convergence_depth_capture),
        'tail_state_check_recorded': "'convergence_depth_tail_state_check'" in child_src,
        'cost_breakdown_fields': {'investment_cost_eur', 'total_planning_cost_eur', 'leak_detector'}
                                 <= set(H.S56_COST_BREAKDOWN_FIELDS),
        'leak_loss_price_written': "'priced_loss_eur'" in inspect.getsource(H.s56_leak_detector),
        'investment_in_entry': "fields['s56_investment'] = facts" in inspect.getsource(H.s56_freeze_entry),
        'pricing_fields': set(H.S56_PRICING_FIELDS) >= {'sigma', 'interface', 'ess', 'esso_slacks', 'closure',
                                                        'settlement_blocks', 'efficiency'},
        'pricing_written_in_child': 'write_s56_pricing_terminal(planning, models, state, eval_dir)' in child_src,
        'settled_stop_summary_recorded': 's56_certification_stop_summary' in inspect.getsource(
            H.s56_settled_stop_record_fields),
        'equality_tolerance_anchor': __import__('definitions').EQUALITY_TOLERANCE == MAPPING_PARAMETERS['tau_cl_pu']
                                     == MAPPING_PARAMETERS['esso_slack_tol'],
        # mapping v2, `solve_profile_reconciliation`: its inputs, checked in the source text as committed (no model
        # import): the record carries the failure summary; the runner writes the ESSO events file and counts them;
        # production's ESSO event records the tier-2 attempt and the success of the loaded solution
        'network_failures_summary_in_record': "'network_failures_summary'" in rec_src,
        'esso_recovery_events_written': all(t in _code_text('p515_g_g1_g4_admm_gates.py') for t in (
            "basename.replace('network_failures_', 'esso_recovery_events_', 1)",
            "'n_esso_recovery_events': len(final_esso_events)", "'retry_solves_credited': _retries",
            "'reconciliation_supported': _supported")),
        'esso_recovery_event_fields': all(t in _code_text('shared_energy_storage_data.py') for t in (
            "'recovery_succeeded': solver_result_succeeded(result)", "'tier2_attempted': tier2_attempted",
            "'tier': 'tier2' if tier2_attempted else 'tier1'", "log_suffix='recovery'",
            "log_suffix='recovery_tier2'")),
    }
    if hz is not None:
        st, ex = hz['by_depth']['settled']['entry_options'], hz['by_depth']['exit']['entry_options']
        checks['settled_declares_stop_disabled'] = st.get(H.S56_SETTLED_STOP_DISABLED_KEY) is True
        checks['settled_declares_pricing_capture'] = st.get(H.S56_PRICING_CAPTURE_KEY) is True
        checks['exit_declares_neither'] = (ex.get(H.S56_SETTLED_STOP_DISABLED_KEY) is None
                                           and ex.get(H.S56_PRICING_CAPTURE_KEY) is None)
        for d in P.DEPTHS:
            tail = (hz['by_depth'][d]['configuration'].get('convergence_depth_tail') or {})
            checks[f'{d}_tail_enabled'] = tail.get('enabled') is True
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise Refused(f'mapping capture path missing: {failed}')
    return checks


# ======================================================================================================================
#  cache roots and the start pool (S56_PLAN_v4_1.md section 4: one search per price level from the best start; the
#  start pool is x0, the reference unit, the recomputed historical plans and the confirmed screen candidates)
# ======================================================================================================================
def build_cache_index(oracle, lattice, data_root, roots, domain=None):
    """At freeze: every eval dir under each declared cache root whose record's eval key equals the key this search
    would request for (z, depth) is indexed (z from the candidate key over the domain). Everything else is listed as
    ignored with its reason. -> [{'root', 'entries': [{'z', 'depth', 'eval_dir', 'eval_key', 'record_sha256'}],
    'ignored': [...]}]"""
    by_key = {lattice.key(z): z for z in (domain if domain is not None else enumerate_domain(lattice))}
    out = []
    for rel in roots:
        absr = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
        if not os.path.isdir(absr):
            raise Refused(f'cache root {rel} does not exist')
        entries, ignored = [], []
        for d, _dirs, fnames in sorted(os.walk(absr)):
            if 'evaluation_record.json' not in fnames or os.path.basename(os.path.dirname(d)) != 'evals':
                continue
            rp = os.path.join(d, 'evaluation_record.json')
            rec = load_json(rp)
            z = by_key.get(rec.get('candidate_key'))
            if z is None:
                ignored.append({'eval_dir': os.path.relpath(d, data_root), 'reason': 'candidate not in the domain'})
                continue
            hit = [dp for dp in P.DEPTHS if oracle.eval_key(z, dp) == rec.get('eval_key')]
            if len(hit) != 1:
                ignored.append({'eval_dir': os.path.relpath(d, data_root),
                                'reason': 'eval key is not this search\'s key at either depth (another configuration)'})
                continue
            if not os.path.basename(d).startswith(rec['eval_key'][:16]):
                ignored.append({'eval_dir': os.path.relpath(d, data_root), 'reason': 'dir name != eval key'})
                continue
            entries.append({'z': list(z), 'depth': hit[0], 'eval_dir': os.path.relpath(d, data_root),
                            'eval_key': rec['eval_key'], 'record_sha256': sha256_file(rp)})
        keys = [(tuple(e['z']), e['depth']) for e in entries]
        if len(set(keys)) != len(keys):
            raise Refused(f'cache root {rel}: a (z, depth) is indexed twice')
        out.append({'root': os.path.relpath(absr, data_root), 'entries': entries, 'ignored': ignored})
    return out


def load_cache(spec, oracle, data_root):
    """At every launch: the frozen cache index re-verified (record hash, key recomputed) and mapped. -> records."""
    records = {}
    for src in spec.get('cache_sources') or []:
        for e in src['entries']:
            z, depth = tuple(e['z']), e['depth']
            ed = os.path.join(data_root, e['eval_dir'])
            if oracle.eval_key(z, depth) != e['eval_key']:
                raise Refused(f"cache {e['eval_dir']}: eval key does not recompute")
            summ, rfile = oracle.summary_from_dir(ed, z, depth)
            if sha256_file(rfile) != e['record_sha256']:
                raise Refused(f"cache {e['eval_dir']}: record changed since the freeze")
            if (z, depth) in records:
                raise Refused(f'cache: {list(z)} {depth} in two cache roots')
            records[(z, depth)] = summ
    return records


def best_start(lattice, pool, cache_records):
    """The plan v2 / v4.1 rule: the start is the pool member with the lowest exit F among the members whose cached
    exit summary is certified (ties by label). Every member must have a cached exit record. -> (z, report)."""
    rows, missing = [], []
    for m in pool:
        z = tuple(m['z'])
        s = cache_records.get((z, P.DEPTH_EXIT))
        if s is None:
            missing.append(m['name'])
            continue
        rows.append({'name': m['name'], 'z': list(z), 'label': lattice.label(z), 'status': s.get('status'),
                     'F': s.get('F'), 'bar': s.get('bar')})
    if missing:
        raise Refused(f'start pool members without a cached exit record (evaluate them first): {missing}')
    cand = sorted((r for r in rows if r['status'] == 'certified' and _fin(r['F']) and _fin(r['bar'])),
                  key=lambda r: (r['F'], r['label']))
    if not cand:
        raise Refused('no start pool member has a certified exit record')
    best = cand[0]
    rep = {'rule': 'argmin exit F over the certified pool members (ties by lattice label)', 'members': rows,
           'excluded_uncertified': [r['name'] for r in rows if r not in cand], 'chosen': best['name']}
    if len(cand) > 1:
        e = P.exit_error(best['bar'], cand[1]['bar'])
        rep['runner_up'] = {'name': cand[1]['name'], 'margin': cand[1]['F'] - best['F'], 'e': e,
                            'within_e': cand[1]['F'] - best['F'] <= e}
    return tuple(best['z']), rep


# ======================================================================================================================
#  real-launch preflight (memory, harness preconditions) and the wall-time projection
# ======================================================================================================================
GIB = 1 << 30
MEMORY_RULE = ('hw.memsize - (wired + anonymous + compressor-occupied pages) x page size >= concurrency x the declared '
               'per-slot GiB (the rule of p515_s55_e1_launcher.memory_preflight, ~958-979, with the per-slot figure '
               'declared in the search / pilot spec instead of the S47 constant)')


def memory_preflight(concurrency, per_slot_gib, vm_stat_text=None, memsize=None):
    """`vm_stat_text` / `memsize` injectable for the tests (zero-solve)."""
    text = vm_stat_text if vm_stat_text is not None else subprocess.run(['vm_stat'], capture_output=True, text=True,
                                                                        check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    need = ('Pages wired down', 'Anonymous pages', 'Pages occupied by compressor')
    total = memsize if memsize is not None else int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True,
                                                                   text=True, check=True).stdout)
    if not isinstance(per_slot_gib, (int, float)) or isinstance(per_slot_gib, bool) or per_slot_gib <= 0:
        return {'pass': False, 'reason': f'per-slot GiB not declared: {per_slot_gib!r}', 'rule': MEMORY_RULE}
    if any(raw.get(k) is None for k in need):
        return {'pass': False, 'missing': [k for k in need if raw.get(k) is None], 'rule': MEMORY_RULE}
    avail = total - sum(raw[k] for k in need) * page
    req = int(concurrency) * float(per_slot_gib) * GIB
    return {'utc': _utc(), 'concurrency': int(concurrency), 'per_slot_gib': float(per_slot_gib),
            'available_gib': avail / GIB, 'required_gib': req / GIB, 'pass': avail >= req, 'rule': MEMORY_RULE,
            'hw_memsize_gib': total / GIB, 'pages': {k: raw.get(k) for k in need}, 'page_size': page}


DRIVER_CLEAN_FILES = ('p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s56_neighbourhood.py',
                      'p516_lattice.py', 'p515_s56_ev_harness_tests.py', 'p515_s56_ev_injection.py',
                      'p515_s56_cohort_diagnostics.py', 'definitions.py')


def real_launch_preflight(H, launch_dir, concurrency, per_slot_gib, data_root, mapping=None):
    """Before any REAL harness launch (zero solves): the mapping parameters final, the harness preconditions
    (`check_campaign_preconditions`: legacy lock, live campaign processes, write-once root, production / harness /
    driver files clean in git), NLP_SOLVER_PATH, and the memory rule. -> the record; raises Refused naming every
    failed check."""
    fails = []
    mapping = mapping if mapping is not None else MAPPING_PARAMETERS
    if mapping.get('provisional'):
        fails.append(f"mapping parameters still provisional (Planner to confirm): {mapping['provisional']}")
    pre = H.check_campaign_preconditions(launch_dir, extra_clean_files=DRIVER_CLEAN_FILES,
                                         lock_path=os.path.join(data_root, '.p515_s56_precheck_no_such_lock'))
    fails += [f'harness precondition: {f}' for f in pre]
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        fails.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    mem = memory_preflight(concurrency, per_slot_gib)
    if not mem.get('pass'):
        fails.append(f'memory preflight: {mem}')
    rec = {'utc': _utc(), 'harness_preconditions': pre, 'memory': mem, 'refusals': fails,
           'note': 'the campaign lock is held by this driver (O_EXCL) when this runs; the lock-file check of '
                   'check_campaign_preconditions is pointed at an absent path for that reason'}
    if fails:
        raise Refused(fails)
    return rec


KAPPA_CONTENTION = 1.18   # gate F (P5_15_S56_CONCURRENCY_GATE_REPORT.md: P2 held 1.183x)


def project_wall_time(t_exit_h, t_set_h, psi, n_blocks, n_confirmation_batches, kappa=KAPPA_CONTENTION):
    """The design's T ~ kappa [psi t_exit sum(blocks) + t_set (confirmation batches)], hours."""
    for name, v in (('t_exit_h', t_exit_h), ('t_set_h', t_set_h), ('psi', psi)):
        if not _fin(v) or v <= 0:
            raise Refused(f'{name} must be a measured positive number (from the pilot), got {v!r}')
    return {'T_h': kappa * (psi * t_exit_h * n_blocks + t_set_h * n_confirmation_batches), 'kappa': kappa,
            'psi': psi, 't_exit_h': t_exit_h, 't_set_h': t_set_h, 'n_blocks': n_blocks,
            'n_confirmation_batches': n_confirmation_batches,
            'formula': 'T = kappa [psi t_exit sum(blocks) + t_set (confirmation batches)]'}


def count_projection(moves, final, s0=1, block=P.BLOCK):
    """The design's count: moves = [{'n', 'p', 'c'}] per accepted move j (n_j = |N_j| - cache hits, p_j = position of
    the first confirmed improver among the uncached, c_j confirmations), final = {'n', 'c'}.
    N_search = S_0 + sum_j (E_j + c_j) + E_f + c_f, E_j = min(n_j, B ceil(p_j / B)), E_f = n_f; blocks =
    S_0 + sum ceil(E / B); confirmation batches = sum c."""
    import math as _m
    rows, total, blocks, conf = [], s0, s0, 0
    for j, m in enumerate(moves, 1):
        e = min(m['n'], block * _m.ceil(m['p'] / block))
        rows.append({'j': j, 'n': m['n'], 'p': m['p'], 'E': e, 'c': m['c']})
        total += e + m['c']
        blocks += _m.ceil(e / block)
        conf += m['c']
    total += final['n'] + final['c']
    blocks += _m.ceil(final['n'] / block)
    conf += final['c']
    return {'N_search': total, 'blocks': blocks, 'confirmation_batches': conf, 'per_move': rows, 'final': final,
            'S_0': s0}


def projection_from_state(state):
    """Blocks and confirmation batches of a finished (dry-run / synthetic) search state, from its evaluated list."""
    acts = {}
    for e in state['evaluated']:
        acts.setdefault((e['action_index'], e['depth']), 0)
        acts[(e['action_index'], e['depth'])] += 1
    blocks = sum(1 for (_a, d) in acts if d == P.DEPTH_EXIT)
    conf = sum(1 for (_a, d) in acts if d == P.DEPTH_SETTLED)
    return {'blocks': blocks, 'confirmation_batches': conf, 'evaluations': state['budget_used']}


# ======================================================================================================================
#  THE PILOT (S56_SEARCH_DRIVER_DESIGN.md, "Pilot protocol"; critical-path step 4). BUILT, NOT LAUNCHED here.
# ======================================================================================================================
#  Seven evaluations at 2x, every one stop-disabled to K_PILOT = 300 with the terminal pricing capture: x0 and the
#  reference unit from three initialisations each, plus one staged plan; in blocks of 5 (one block = one attached launch
#  = one harness campaign root; exit 4 while blocks remain). The exit value of each run is read at its would-be
#  certification cycle; the settled value at K. `pilot_outputs` (zero solves) computes, from the seven traces: single
#  offsets and their sign, pair offsets over the 3 x 3 initialisation combinations and the staged plan against both,
#  the dF_exit and dF_set spreads, V_pair beside, the bars, t_exit, t_set, psi, the leak detector at the settled point,
#  and K_EV and e_EV by the frozen rule:
#    K_EV = the smallest K >= max_run c* (every run past its would-be certification cycle) at which S_run(K) <= s_share
#           for every run (S = range of F over the last W cycles); whether it then holds at every later K is reported;
#    e_EV = max(2 x the largest pair offset |dF_exit - dF_set| (dF_set at K_EV), the dF_exit spread over the 3 x 3
#           combinations, the largest bar sum).
#  INITIALISATIONS (FINAL, integration ruling 3, 601775c0): the existing keyed harness option `rho0_scale` (W183a:
#  rho0 x s on every channel of the TSO and DSOs and the ESSO rho), s in (1.0, 0.5, 2.0); 1.0 is the default run (no
#  option written). Staged plan (ruling 3): the memory probe's SRP1-grid plan
#  n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5 (asserted against the lattice label).
#  s_share (FINAL, integration ruling 2): S(x) <= 500 EUR per evaluation (1,000 EUR of the pair's 2,500 EUR); K_EV is
#  the smallest K at which every pilot run's S <= 500 EUR and K >= every run's would-be certification cycle. It is
#  frozen into the pilot spec and read from it by --pilot-evaluate (a different --s-share-eur is refused).
#  memory_per_slot_gib (FINAL, ruling 7): 4.0 GiB for EV evaluations (measured 2.43 GiB at 3 cycles; step 4 children
#  <= 3.53 GiB over full runs); the pilot measures the full-run peak; the search spec then uses that peak x 1.15.
PILOT_SCHEMA = 'p515_s56_pilot_spec_v1'
PILOT_RESULT_SCHEMA = 'p515_s56_pilot_result_v1'
K_PILOT = 300
PILOT_FLEX = 2.0
PILOT_BLOCK = 5
PILOT_INIT_VARIANTS = (('i1', 1.0), ('i2', 0.5), ('i3', 2.0))
PILOT_INIT_VARIANTS_PROPOSED = PILOT_INIT_VARIANTS   # the integration build's name, kept (ruling 3 confirmed the values)
PILOT_GRIDS = ('srp1grid', 'five_block')
PILOT_STAGED_LABEL = {'srp1grid': 'n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5'}
S_SHARE_EUR = 500.0                 # integration ruling 2
MEMORY_PER_SLOT_GIB_EV = 4.0        # integration ruling 7
PILOT_PREDICTION_IDS = ('PP1', 'PP2', 'PP3', 'PP4', 'PP5', 'PP6', 'PP7', 'PP8')
RULINGS_REF = 'S56_SEARCH_DRIVER_DESIGN.md, Integration rulings 1-7 (601775c0)'


def _evt():
    import p515_s56_ev_harness_tests as EVT   # declarations of the two EV twins and the step 4 spec-like configuration
    return EVT


def pilot_entries(H, grid, init_variants=PILOT_INIT_VARIANTS, k_pilot=K_PILOT):
    """The seven entries in launch order (x0 / unit / staged at the first initialisation, then the others)."""
    EVT = _evt()
    _derived, ev = EVT.ev_declaration(grid)
    base = {H.S56_COHORT_FORMULATION_KEY: 'independent_units_all_node_years', H.S56_TV_IN_OBJECTIVE_KEY: True,
            H.S56_ESSO_SLACK_PENALTY_KEY: 1e4, H.S56_EV_PROFILES_KEY: dict(ev), 'flex_price_multiplier': PILOT_FLEX,
            H.S56_SETTLED_STOP_DISABLED_KEY: True, H.S56_PRICING_CAPTURE_KEY: True}
    plans = {'x0': EVT.X0_STAGED, 'unit': EVT.REF_UNIT, 'staged': EVT.STAGED[grid]}
    out = []
    for i, (iname, scale) in enumerate(init_variants):
        roles = ('x0', 'unit', 'staged') if i == 0 else ('x0', 'unit')
        for role in roles:
            opts = dict(copy.deepcopy(base), **{H.S56_STAGED_INVESTMENT_KEY: copy.deepcopy(plans[role])})
            if scale != 1.0:
                opts['rho0_scale'] = float(scale)
            out.append({'label': f'{role}_{iname}', 'role': role, 'init': iname, 'rho0_scale': float(scale),
                        'options': opts})
    if len(out) != 7:
        raise Refused(f'the pilot needs 7 evaluations, got {len(out)}')
    return out


def pilot_capture_checklist(H):
    """Rule eleven for the pilot (zero solves): a capture path exists in the committed harness / driver for every
    quantity the pilot's outputs and recorded predictions read. -> checklist; raises Refused naming the failed checks."""
    import inspect
    rec_src = inspect.getsource(H.build_evaluation_record)
    checks = {
        'mapping_capture_paths': bool(assert_mapping_capture_paths(H)),
        'per_cycle_F_terms (PP3, PP4, PP5, K_EV, e_EV)': {'recourse', 'gross_operational_cost',
                                                         'terminal_salvage_value'} <= set(H.PER_CYCLE_TRAJECTORY_FIELDS),
        'would_be_certification_cycle (PP1, PP4)': {'consecutive_converged_cycles', 'cycle_convergence'}
                                                    <= set(H.PER_CYCLE_TRAJECTORY_FIELDS),
        'local_solve_failures_in_record (PP1, PP2)': "'local_solve_failures'" in rec_src,
        'cycles_run_and_cap_in_record (PP1, PP2)': "'cycles_run'" in rec_src and "'cap'" in rec_src,
        'tail_depth_per_cycle (PP2)': 'CONVERGENCE_DEPTH_TAIL_STATE_FILE' in inspect.getsource(
            H.persist_convergence_depth_capture),
        'leak_priced_loss (PP6)': "'priced_loss_eur'" in inspect.getsource(H.s56_leak_detector),
        'peak_rss_in_record (PP7)': "'peak_rss'" in rec_src and 'child_python_process_ru_maxrss' in inspect.getsource(H),
        'per_cycle_wall (PP8)': 'cycle_wall_s' in H.PER_CYCLE_RESPONSE_FIELDS,
        'wall_time_in_record (PP8, t_exit / t_set)': "'wall_time_s'" in rec_src,
    }
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise Refused(f'pilot capture path missing: {failed}')
    return checks


def validate_predictions(pred):
    """A real pilot's recorded predictions: every one of PP1-PP8, each with its statement, its operational
    evaluation (the formula), its threshold(s) and its failure consequence."""
    if not isinstance(pred, dict) or not isinstance(pred.get('predictions'), dict):
        raise Refused('the predictions file must hold {"predictions": {...}}')
    missing = [k for k in PILOT_PREDICTION_IDS if k not in pred['predictions']]
    bad = [k for k, v in pred['predictions'].items()
           if not all(isinstance(v.get(f), str) and v[f] for f in ('statement', 'evaluated_as', 'on_failure'))]
    if missing or bad:
        raise Refused(f'predictions incomplete: missing {missing}; without statement / evaluated_as / on_failure {bad}')
    return pred


def pilot_spec(*, pilot_id, grid, oracle, test_only, data_root, memory_per_slot_gib=None,
               init_variants=PILOT_INIT_VARIANTS, k_pilot=K_PILOT, predictions=None, grid_note=None):
    import p515_s44_campaign_harness as H
    if grid not in PILOT_GRIDS:
        raise Refused(f'grid {grid!r} not in {PILOT_GRIDS} (the author confirms the grid after the memory probe)')
    if oracle not in ('stub_children', 'harness'):
        raise Refused(f'pilot oracle {oracle!r}')
    if oracle == 'stub_children' and not test_only:
        raise Refused('a stub-children pilot must be test-only')
    capture = None
    if memory_per_slot_gib is None:
        memory_per_slot_gib = MEMORY_PER_SLOT_GIB_EV
    if oracle == 'harness':
        if MAPPING_PARAMETERS.get('provisional') and not test_only:
            raise Refused(f"mapping parameters still provisional (Planner to confirm): {MAPPING_PARAMETERS['provisional']}")
        if not _fin(memory_per_slot_gib) or memory_per_slot_gib <= 0:
            raise Refused('a real pilot must declare memory_per_slot_gib (from the memory probe)')
        if not test_only:
            if memory_per_slot_gib != MEMORY_PER_SLOT_GIB_EV:
                raise Refused(f'memory_per_slot_gib {memory_per_slot_gib} != {MEMORY_PER_SLOT_GIB_EV} (ruling 7)')
            if [list(v) for v in init_variants] != [list(v) for v in PILOT_INIT_VARIANTS] or k_pilot != K_PILOT:
                raise Refused('a real pilot uses the ruled initialisations and K_pilot (rulings 3; Oracle mapping)')
            validate_predictions(predictions)
        capture = pilot_capture_checklist(H)
    EVT = _evt()
    derived, _ev = EVT.ev_declaration(grid)
    cfg, required = EVT._spec_like(derived, cap=k_pilot)
    entries = pilot_entries(H, grid, init_variants, k_pilot)
    case_rel = H.s56_case_rel(H.validate_derived_instance(derived))
    for e in entries:
        canon = H.s56_staged_canonical(e['options'][H.S56_STAGED_INVESTMENT_KEY], case_rel)[0]
        e['candidate_key'] = H.candidate_key(canon)
        e['eval_key'] = compose_eval_key(H, cfg, e['options'], k_pilot, e['candidate_key'])
    staged_label = None
    if grid in PILOT_STAGED_LABEL:   # ruling 3: the staged plan is named; its lattice label must be that name
        lat = build_lattice(data_root)[0]
        st = next(e for e in entries if e['role'] == 'staged')
        z = lat.z_of_canonical(H.s56_staged_canonical(st['options'][H.S56_STAGED_INVESTMENT_KEY], case_rel)[0])
        staged_label = lat.label(tuple(z)) if z is not None else None
        if staged_label != PILOT_STAGED_LABEL[grid] or lat.key(tuple(z)) != st['candidate_key']:
            raise Refused(f'staged plan {staged_label!r} / key is not the ruled {PILOT_STAGED_LABEL[grid]!r}')
    labels = [e['label'] for e in entries]
    blocks = [labels[i:i + PILOT_BLOCK] for i in range(0, len(labels), PILOT_BLOCK)]
    return {'schema': PILOT_SCHEMA, 'pilot_id': pilot_id, 'frozen_utc': _utc(), 'grid': grid, 'test_only': test_only,
            'grid_note': grid_note,
            'oracle': {'kind': oracle}, 'k_pilot': k_pilot, 'flex_price_multiplier': PILOT_FLEX,
            'configuration': cfg, 'required_consecutive_cycles': required, 'entries': entries, 'blocks': blocks,
            'staged_plan_label': staged_label,
            'memory_per_slot_gib': memory_per_slot_gib, 'mapping': MAPPING_PARAMETERS,
            'init_variants': {'values': [list(v) for v in init_variants], 'option': 'rho0_scale (existing, keyed)',
                              'status': f'FINAL ({RULINGS_REF}, ruling 3)'},
            'rules': {'K_EV': 'smallest K >= max c* with S_run(K) <= s_share for every run (pilot_outputs)',
                      'e_EV': 'max(2 x max pair offset at K_EV, dF_exit spread over 3 x 3, max bar sum)',
                      's_share_eur': S_SHARE_EUR,
                      's_share': f'S(x) <= {S_SHARE_EUR:g} EUR per evaluation ({RULINGS_REF}, ruling 2)',
                      'memory_per_slot_gib': f'{MEMORY_PER_SLOT_GIB_EV:g} GiB per EV slot ({RULINGS_REF}, ruling 7); '
                                             'the search spec uses the pilot\'s measured peak x 1.15'},
            'capture_path_checklist': capture,
            'predictions_recorded_before_run': predictions,
            'authority': AUTHORITY + ['S56_SEARCH_DRIVER_DESIGN.md: Pilot protocol; Oracle mapping (acc80a2e)',
                                      RULINGS_REF],
            'code_sha256': code_hashes(), 'git_head': _git_head(data_root)}


def pilot_freeze(root, spec):
    if os.path.exists(root) and [n for n in os.listdir(root) if not n.endswith('.log')]:
        raise Refused(f'pilot root exists and holds files other than launch logs (write-once): {root}')
    os.makedirs(root, exist_ok=True)
    text = P.canonical_json(spec)
    digest = sha256_text(text)
    path = os.path.join(root, f'pilot_spec_{digest[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    return path, digest


def load_pilot_spec(root):
    names = sorted(n for n in os.listdir(root) if n.startswith('pilot_spec_') and n.endswith('.json'))
    if len(names) != 1:
        raise Refused(f'expected exactly one pilot_spec_*.json in {root}, found {names}')
    path = os.path.join(root, names[0])
    digest = sha256_file(path)
    if names[0] != f'pilot_spec_{digest[:8]}.json':
        raise Refused(f'{names[0]} does not carry its own hash prefix')
    spec = load_json(path)
    if spec.get('schema') != PILOT_SCHEMA:
        raise Refused(f'pilot spec schema {spec.get("schema")!r}')
    return path, digest, spec


def pilot_run(root, *, data_root, lock_path, take_over_stale=False, preflight_only_out=None):
    """ONE pilot launch = the next block of <= 5 (attached; exit 4 while blocks remain, 0 when every block is done).
    `preflight_only_out` (a NEW directory): zero solves -- the lock, the chain check of completed blocks and
    `real_launch_preflight` for the next block, exactly as a launch runs them before it writes anything; the record is
    written to that directory, the lock released, nothing launched, nothing written under the pilot root."""
    import p515_s44_campaign_harness as H
    _sp, sha, spec = load_pilot_spec(root)
    stub = spec['oracle']['kind'] == 'stub_children'
    if stub and not spec.get('test_only'):
        raise Refused('stub pilot without test_only')
    if preflight_only_out is not None:
        if stub:
            raise Refused('--preflight-only is for a real pilot')
        if os.path.exists(preflight_only_out):
            raise Refused(f'--out exists (write-once): {preflight_only_out}')
    idx0, complete0 = _complete_indices(root)
    if preflight_only_out is not None and set(idx0) != set(complete0):
        raise Refused(f'an incomplete launch exists ({sorted(set(idx0) - set(complete0))}): a real launch would move '
                      'it aside first; the preflight-only mode moves nothing')
    lock, _moved = acquire_lock(lock_path, spec['pilot_id'], len(complete0) + 1, root, take_over_stale)
    try:
        if preflight_only_out is not None:
            return _pilot_preflight_only(root, spec, sha, lock, data_root, H, preflight_only_out)
        return _pilot_launch(root, spec, sha, stub, lock, lock_path, data_root, H)
    finally:
        release_lock(lock_path)


def _pilot_chain(root, spec, sha, complete):
    prev = None
    for n in complete:   # the chain and every frozen key of every completed block
        _ld, rec_p, man_p = launch_paths(root, n)
        lr, man = load_json(rec_p), load_json(man_p)
        if lr.get('previous_launch_manifest_sha256') != prev or lr.get('pilot_spec_sha256') != sha:
            raise Refused(f'pilot launch {n}: hash chain / spec hash')
        if any(sha256_file(os.path.join(root, f)) != h for f, h in man.items()):
            raise Refused(f'pilot launch {n}: a manifest hash does not hold')
        want = {e['label']: e['eval_key'] for e in spec['entries'] if e['label'] in spec['blocks'][n - 1]}
        if {r['label']: r['eval_key'] for r in lr['records']} != want:
            raise Refused(f'pilot launch {n}: records are not the frozen block')
        prev = sha256_file(man_p)
    return prev


def _pilot_preflight_only(root, spec, sha, lock, data_root, H, out_dir):
    _idx, complete = _complete_indices(root)
    prev = _pilot_chain(root, spec, sha, complete)
    n = len(complete) + 1
    if n > len(spec['blocks']):
        raise Refused('pilot: every block done; nothing to preflight')
    block = spec['blocks'][n - 1]
    ldir = launch_paths(root, n)[0]
    rec = {'schema': 'p515_s56_pilot_preflight_only_v1', 'utc': _utc(), 'pilot_root': os.path.relpath(root, data_root),
           'pilot_spec_sha256': sha, 'next_launch_index': n, 'next_block': block,
           'previous_launch_manifest_sha256': prev, 'launch_dir_would_be': os.path.relpath(ldir, data_root),
           'lock_acquired': lock, 'git_head': _git_head(data_root), 'code_sha256': code_hashes(),
           'solves': 0, 'launched': False}
    try:
        rec['real_launch_preflight'] = real_launch_preflight(H, ldir, len(block), spec['memory_per_slot_gib'],
                                                             data_root, mapping=spec['mapping'])
        rec['pass'] = True
    except Refused as error:
        rec['real_launch_preflight'] = {'refusals': error.reasons}
        rec['pass'] = False
    rec['launch_dir_created'] = os.path.exists(ldir)
    os.makedirs(out_dir)
    rsha = write_once_json(os.path.join(out_dir, 'pilot_preflight.json'), rec)
    _log(f"pilot preflight-only (block {n}: {block}): pass {rec['pass']}; record sha256 {rsha}")
    if not rec['pass']:
        raise Refused(rec['real_launch_preflight']['refusals'])
    return EXIT_TERMINATED


def _pilot_launch(root, spec, sha, stub, lock, lock_path, data_root, H):
    handle_interruptions(root, spec, spec['oracle']['kind'])
    _idx, complete = _complete_indices(root)
    prev = _pilot_chain(root, spec, sha, complete)
    n = len(complete) + 1
    if n > len(spec['blocks']):
        _log('pilot: every block done')
        return EXIT_TERMINATED
    block = spec['blocks'][n - 1]
    ldir, rec_p, man_p = launch_paths(root, n)
    preflight = None
    if not stub:
        preflight = real_launch_preflight(H, ldir, len(block), spec['memory_per_slot_gib'], data_root,
                                          mapping=spec['mapping'])
    os.makedirs(ldir)
    ents = [e for e in spec['entries'] if e['label'] in block]
    cands = [(e['label'], None, copy.deepcopy(e['options'])) for e in ents]
    cfg = dict(copy.deepcopy(spec['configuration']), name=f"{spec['pilot_id']} block {n}")
    spec_path, csha, hspec = H.freeze_campaign_spec(
        ldir, f"{spec['pilot_id']}_l{n:03d}", cands, cfg, spec['k_pilot'], len(block), spec['authority'],
        required_consecutive_cycles=spec['required_consecutive_cycles'],
        extra={'test_only_stub': stub, 's56_pilot': {'pilot_id': spec['pilot_id'], 'block': n}})
    by = {e['label']: e for e in hspec['candidates']}
    bad = [e['label'] for e in ents if by[e['label']]['eval_key'] != e['eval_key']]
    if bad:
        raise Refused(f'pilot block {n}: harness-frozen eval keys differ from the pilot spec: {bad}')
    content = set_lock_spec(lock_path, lock, csha)
    if stub:
        ctx = stub_context(H, ldir, spec_path, csha, hspec, n, lock_path, spec.get('test_only'))
    else:
        if os.path.abspath(lock_path) != os.path.abspath(H.CAMPAIGN_LOCK_PATH):
            raise Refused('real pilot children verify the default campaign lock')
        ctx = H.CampaignContext(ldir, spec_path, csha, hspec, log=_log)
    t0 = time.time()
    H.evaluate(block, ctx, heartbeat_tag=f'pilot_{n:03d}')
    recs = []
    for lab in block:
        ed = os.path.join(ldir, 'evals', by[lab]['eval_dir'])
        rp = os.path.join(ed, 'evaluation_record.json')
        recs.append({'label': lab, 'eval_key': by[lab]['eval_key'], 'eval_dir': os.path.relpath(ed, root),
                     'record_sha256': sha256_file(rp) if os.path.isfile(rp) else None,
                     'exit_code': (open(os.path.join(ed, 'exit_code.txt')).read().strip()
                                   if os.path.isfile(os.path.join(ed, 'exit_code.txt')) else None)})
    lr = {'schema': 'p515_s56_pilot_launch_v1', 'launch_index': n, 'pilot_spec_sha256': sha,
          'previous_launch_manifest_sha256': prev, 'block': block, 'records': recs, 'wall_s': time.time() - t0,
          'lock': content, **({'real_launch_preflight': preflight} if preflight else {})}
    rsha = write_once_json(rec_p, lr)
    manifest = {os.path.relpath(f, root): sha256_file(f) for f in files_under(ldir)}
    manifest[os.path.relpath(rec_p, root)] = rsha
    write_once_json(man_p, manifest)
    _log(f'pilot block {n}/{len(spec["blocks"])} done: {block}')
    return EXIT_SUSPENDED if n < len(spec['blocks']) else EXIT_TERMINATED


def trace_from_files(files, label, role, init, params=MAPPING_PARAMETERS):
    """One pilot run's trace from its real eval dir (`read_eval_files`)."""
    rec, rows = files['record'], files['rows']
    inv, F, _bad = f_series(rec, rows)
    tail = {p.get('cycle'): bool(p.get('active')) for p in ((files.get('tail') or {}).get('per_cycle') or [])}
    wall = {r['cycle']: r.get('cycle_wall_s') for r in rows}
    child_s = (rec.get('wall_time_s') or {}).get('child_process_s')
    loop_s = sum(w for w in wall.values() if _fin(w))
    return {'label': label, 'role': role, 'init': init, 'I': inv, 'F': F,
            'consec': {r['cycle']: r.get('consecutive_converged_cycles') for r in rows},
            'conv': {r['cycle']: r.get('cycle_convergence') for r in rows}, 'tail': tail, 'wall': wall,
            'overhead_s': (child_s - loop_s) if _fin(child_s) else None, 'K': rec.get('cap'),
            'required': int((rec.get('certification') or {}).get('required_consecutive_cycles') or 10),
            'terminal': map_record(files, P.DEPTH_SETTLED, params)}


def synthetic_trace(label, role, init, seed, K):
    """A deterministic synthetic trace for the stub dry run (NOT a model of the system)."""
    def u(tag):
        return int(sha256_text(f'{seed}|{label}|{tag}')[:12], 16) / float(1 << 48)
    f_inf = 873.9e6 + {'x0': 0.0, 'unit': 3000.0, 'staged': -20000.0}[role] + (u('lim') - 0.5) * 400.0
    amp, tau, c0 = 2.0e6 * (0.5 + u('amp')), 8.0 + 6.0 * u('tau'), 40 + int(40 * u('c0'))
    F, consec, conv, tail, wall = {}, {}, {}, {}, {}
    run = 0
    for k in range(1, K + 1):
        F[k] = f_inf + amp * math.exp(-k / tau) * math.cos(k / 3.0) + 50.0 * (u(f'n{k}') - 0.5) * math.exp(-k / 120.0)
        cv = k >= c0
        run = run + 1 if cv else 0
        conv[k], consec[k], tail[k] = cv, run, (k - 1) >= c0
        wall[k] = 25.0 + 5.0 * u(f'w{k}')
    term = {'depth': P.DEPTH_SETTLED, 'status': SETTLED_STATUS, 'F': F[K],
            'V': {'S': max(F[k] for k in range(K - 9, K + 1)) - min(F[k] for k in range(K - 9, K + 1)),
                  'PR': 100.0 * (u('pr') - 0.5), 'G': 50.0 * u('g'), 'A': 300.0 * u('a')},
            'gates_pass': True, 'leak': {'priced_loss_eur': 30.0 * u('leak'), 'within_ceiling': 30.0 * u('leak') <= 25.0}}
    return {'label': label, 'role': role, 'init': init, 'I': 0.0, 'F': F, 'consec': consec, 'conv': conv, 'tail': tail,
            'wall': wall, 'overhead_s': 30.0, 'K': K, 'required': 10, 'terminal': term, 'synthetic': True}


def pilot_outputs(traces, s_share_eur, W=MAPPING_PARAMETERS['W'], bar_window=MAPPING_PARAMETERS['bar_window'],
                  blocks=None):
    """The pilot's outputs and the frozen K_EV / e_EV rules (section comment). Pure; zero solves."""
    if not _fin(s_share_eur) or s_share_eur <= 0:
        raise Refused(f's_share_eur must be declared (Planner): {s_share_eur!r}')
    runs = {}
    for t in traces:
        c = next((k for k in sorted(t['consec']) if isinstance(t['consec'][k], int)
                  and t['consec'][k] >= t['required']), None)
        bar, _st = bar_at(t['F'], c, bar_window) if c is not None else (None, [])
        cyc = sorted(t['F'])
        S = {K: (max(t['F'][k] for k in range(K - W + 1, K + 1)) - min(t['F'][k] for k in range(K - W + 1, K + 1)))
             for K in cyc if K >= W and all(t['F'].get(k) is not None for k in range(K - W + 1, K + 1))}
        t_exit = (t['overhead_s'] + sum(t['wall'][k] for k in range(1, c + 1))) if c is not None and \
            t['overhead_s'] is not None else None
        runs[t['label']] = {'role': t['role'], 'init': t['init'], 'c_star': c, 'F_exit': t['F'].get(c),
                            'bar_exit': bar, 'S': S, 't_exit_s': t_exit, 'K': t['K'],
                            'leak_settled_point': (t['terminal'] or {}).get('leak'),
                            'terminal_status': (t['terminal'] or {}).get('status'),
                            'terminal_V': (t['terminal'] or {}).get('V'),
                            'terminal_gates_pass': (t['terminal'] or {}).get('gates_pass')}
    out = {'schema': PILOT_RESULT_SCHEMA, 's_share_eur': s_share_eur, 'W': W, 'runs': {}, 'failures': []}
    no_exit = [lab for lab, r in runs.items() if r['c_star'] is None]
    if no_exit:
        out['failures'].append(f'no would-be certification cycle within K: {no_exit}')
        out['runs'] = runs
        return out
    k_lo = max(r['c_star'] for r in runs.values())
    K_max = min(r['K'] for r in runs.values())
    k_ev = next((K for K in range(max(k_lo, W), K_max + 1)
                 if all(r['S'].get(K) is not None and r['S'][K] <= s_share_eur for r in runs.values())), None)
    out['K_EV'] = k_ev
    if k_ev is None:
        out['failures'].append(f'S does not meet s_share {s_share_eur} at any K in [{k_lo}, {K_max}]')
    else:
        out['K_EV_holds_at_every_later_K'] = all(r['S'][K] <= s_share_eur for r in runs.values()
                                                 for K in range(k_ev, K_max + 1) if K in r['S'])
        tr = {t['label']: t for t in traces}
        for lab, r in runs.items():
            t = tr[lab]
            r['F_set'] = t['F'][k_ev]
            r['single_offset'] = r['F_exit'] - r['F_set']
            r['single_offset_sign'] = (r['single_offset'] > 0) - (r['single_offset'] < 0)
            r['t_set_s'] = (t['overhead_s'] + sum(t['wall'][k] for k in range(1, k_ev + 1))
                            if t['overhead_s'] is not None else None)
            last = range(k_ev - W + 1, k_ev + 1)
            r['settled_checks_at_K_EV'] = {'tail_depth_last_W': all(t['tail'].get(k) is True for k in last),
                                           'would_have_certified_last_W': all(t['conv'].get(k) is True for k in last)}
        by_role = {}
        for lab, r in runs.items():
            by_role.setdefault(r['role'], []).append(lab)
        pairs = []
        for a_role, b_role in (('x0', 'unit'), ('x0', 'staged'), ('unit', 'staged')):
            for a in sorted(by_role.get(a_role, [])):
                for b in sorted(by_role.get(b_role, [])):
                    ra, rb = runs[a], runs[b]
                    dfe, dfs = rb['F_exit'] - ra['F_exit'], rb['F_set'] - ra['F_set']
                    pv = None
                    if ra['terminal_V'] and rb['terminal_V']:
                        pv = P.v_pair(rb['terminal_V'], ra['terminal_V'])
                    pairs.append({'pair': f'{b} - {a}', 'kind': f'{b_role}-{a_role}', 'dF_exit': dfe, 'dF_set': dfs,
                                  'offset': abs(dfe - dfs), 'bar_sum': ra['bar_exit'] + rb['bar_exit'],
                                  'V_pair_terminal': pv})
        xu = [p for p in pairs if p['kind'] == 'unit-x0']
        spread_exit = max(p['dF_exit'] for p in xu) - min(p['dF_exit'] for p in xu)
        spread_set = max(p['dF_set'] for p in xu) - min(p['dF_set'] for p in xu)
        max_off = max(p['offset'] for p in pairs)
        max_bar = max(p['bar_sum'] for p in pairs)
        out.update({'pairs': pairs, 'dF_exit_spread_3x3': spread_exit, 'dF_set_spread_3x3': spread_set,
                    'max_pair_offset': max_off, 'max_bar_sum': max_bar,
                    'e_EV': max(2.0 * max_off, spread_exit, max_bar),
                    'e_EV_terms': {'2_x_max_pair_offset': 2.0 * max_off, 'dF_exit_spread': spread_exit,
                                   'max_bar_sum': max_bar}})
        te = [r['t_exit_s'] for r in runs.values() if r['t_exit_s'] is not None]
        ts = [r['t_set_s'] for r in runs.values() if r.get('t_set_s') is not None]
        out['t_exit_h'] = (sum(te) / len(te)) / 3600.0 if te else None
        out['t_set_h'] = (sum(ts) / len(ts)) / 3600.0 if ts else None
        psis = []
        for blk in blocks or []:
            vals = [runs[lab]['t_exit_s'] for lab in blk if runs.get(lab, {}).get('t_exit_s') is not None]
            if len(vals) >= 2:
                psis.append({'block': blk, 'psi': max(vals) / (sum(vals) / len(vals))})
        out['psi_per_block'] = psis
        out['psi'] = max((p['psi'] for p in psis), default=None)
        out['psi_definition'] = ('block makespan / mean t_exit of the block (max over blocks of >= 2 runs); the '
                                 'pilot runs concurrently, so t_exit already carries contention (kappa beside)')
    out['runs'] = {lab: {k: v for k, v in r.items() if k != 'S'} | {'S_at_K_EV': r['S'].get(out.get('K_EV'))}
                   for lab, r in runs.items()}
    return out


def pilot_evaluate(root, out_dir, s_share_eur, data_root):
    """Zero solves: every pilot record (real mapping, or synthetic traces for a stub pilot) -> pilot_result.json.
    s_share is read from the frozen spec (ruling 2); an explicit `s_share_eur` must equal it."""
    _sp, sha, spec = load_pilot_spec(root)
    frozen_share = (spec.get('rules') or {}).get('s_share_eur')
    if frozen_share is not None:
        if s_share_eur is not None and s_share_eur != frozen_share:
            raise Refused(f's_share_eur {s_share_eur} differs from the frozen {frozen_share} (ruling 2)')
        s_share_eur = frozen_share
    _idx, complete = _complete_indices(root)
    if len(complete) != len(spec['blocks']):
        raise Refused(f'pilot incomplete: {len(complete)} of {len(spec["blocks"])} blocks')
    traces = []
    for n in complete:
        lr = load_json(launch_paths(root, n)[1])
        for r in lr['records']:
            e = next(x for x in spec['entries'] if x['label'] == r['label'])
            if spec['oracle']['kind'] == 'stub_children':
                rec = load_json(os.path.join(root, r['eval_dir'], 'evaluation_record.json'))
                if rec.get('status') != 'stub' or rec.get('candidate_key') != e['candidate_key'] or \
                        not os.path.basename(r['eval_dir']).startswith(e['eval_key'][:16]):
                    raise Refused(f"stub record {r['label']} is not the frozen entry")
                traces.append(synthetic_trace(e['label'], e['role'], e['init'], spec['pilot_id'], spec['k_pilot']))
            else:
                files = read_eval_files(os.path.join(root, r['eval_dir']))
                if sha256_file(os.path.join(root, r['eval_dir'], 'evaluation_record.json')) != r['record_sha256']:
                    raise Refused(f"record {r['label']} changed since its launch")
                traces.append(trace_from_files(files, e['label'], e['role'], e['init'], spec['mapping']))
    res = pilot_outputs(traces, s_share_eur, blocks=spec['blocks'])
    res.update({'pilot_spec_sha256': sha, 'grid': spec['grid'], 'oracle': spec['oracle'],
                'synthetic': spec['oracle']['kind'] == 'stub_children', 'utc': _utc(),
                'objective_convention': 'F = I(x) + gross - TV credit, EUR of total discounted planning cost',
                'kappa_for_projection': KAPPA_CONTENTION})
    os.makedirs(out_dir)
    rsha = write_once_json(os.path.join(out_dir, 'pilot_result.json'), res)
    write_once_json(os.path.join(out_dir, 'pilot_result_manifest_sha256.json'), {'pilot_result.json': rsha})
    return res


# ======================================================================================================================
#  freeze
# ======================================================================================================================
def code_drift(spec):
    """Spec v2 (Planner decision P2): the spec's frozen `code_sha256` against the files now -> [{'file', 'frozen',
    'now'}] for every file that differs (empty = no drift)."""
    frozen = spec.get('code_sha256') or {}
    now = code_hashes()
    return [{'file': n, 'frozen': frozen.get(n), 'now': now.get(n)} for n in sorted(set(frozen) | set(now))
            if frozen.get(n) != now.get(n)]


def code_hashes():
    out = {}
    for name in ('p515_s56_search_policy.py', SCRIPT_NAME, 'p515_s56_neighbourhood.py', 'p516_lattice.py',
                 'p515_s44_campaign_harness.py', 'gate_result_io.py'):
        path = os.path.join(CODE_ROOT, name)
        out[name] = sha256_file(path) if os.path.isfile(path) else None
    return out


def _git_head(data_root):
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def build_spec(*, search_id, arm, lattice, instance_rec, start_name, n_max, k_ev, oracle, class_rank, test_only,
               data_root, synthetic_seed=None, synthetic_target=None, cache_sources=(), harness_config=None,
               start_z=None, start_report=None, mapping=None):
    if oracle not in ORACLES:
        raise Refused(f'oracle {oracle!r} not in {ORACLES}')
    if not isinstance(n_max, int) or n_max < 0 or not isinstance(k_ev, int) or k_ev < 1:
        raise Refused('N_max (int >= 0) and K_EV (int >= 1) must be declared at freeze (set after the pilot)')
    if oracle == 'harness':
        mp = mapping if mapping is not None else MAPPING_PARAMETERS
        if mp.get('provisional') and not test_only:
            raise Refused(f"harness oracle: mapping parameters still provisional (Planner to confirm): "
                          f"{mp['provisional']}")
        hc0 = harness_config or {}
        if not _fin(hc0.get('memory_per_slot_gib')) or hc0['memory_per_slot_gib'] <= 0:
            raise Refused('harness oracle: the harness config must declare memory_per_slot_gib (from the memory '
                          'probe) for the real-launch memory preflight')
    z0 = tuple(start_z) if start_z is not None else named_start(lattice, start_name)
    why = lattice.reasons(z0)
    if why:
        raise Refused(f'start {start_name} inadmissible: {why}')
    if tuple(class_rank) not in (P.CLASS_RANK, P.CLASS_RANK_REVERSED):
        raise Refused(f'class rank {class_rank} is neither the default nor the reversed control')
    if oracle != 'harness' and not test_only:
        raise Refused('a synthetic or stub-children oracle needs a test-only spec')
    spec = {
        'schema': SEARCH_SPEC_SCHEMA, 'search_id': search_id, 'arm': arm, 'frozen_utc': _utc(),
        'authority': AUTHORITY, 'test_only': bool(test_only),
        'start': dict({'name': start_name, 'z': list(z0), 'label': lattice.label(z0), 'key': lattice.key(z0)},
                      **({'best_of_pool': start_report} if start_report is not None else {})),
        'policy': {'R': P.R_EUR, 'e_floor': P.E_FLOOR_EUR, 'block': P.BLOCK, 'cap': P.CAP,
                   'class_rank': list(class_rank), 'n_max': n_max, 'k_ev': k_ev,
                   'acceptance': 'plan v4.1 section 3 (P.confirm)', 'classification': 'P.classify_exit'},
        'instance': instance_rec, 'domain_size': DOMAIN_SIZE,
        'oracle': {'kind': oracle},
        'mapping': mapping if mapping is not None else MAPPING_PARAMETERS,
        'cache_sources': list(cache_sources),
        'code_sha256': code_hashes(), 'git_head': _git_head(data_root), 'interpreter': sys.executable,
    }
    if oracle in ('stub_children', 'harness'):
        hc = copy.deepcopy(harness_config or {})
        need = all(isinstance((hc.get('by_depth') or {}).get(d), dict) and
                   {'configuration', 'cap', 'entry_options'} <= set(hc['by_depth'][d]) for d in P.DEPTHS)
        if not need or not isinstance(hc.get('concurrency'), int) or not isinstance(
                hc.get('required_consecutive_cycles'), int):
            raise Refused('a harness oracle needs --harness-config with by_depth.{exit, settled}.{configuration, cap, '
                          'entry_options}, concurrency and required_consecutive_cycles')
        if hc['by_depth']['settled']['cap'] != k_ev:
            raise Refused(f"the settled cap {hc['by_depth']['settled']['cap']} != K_EV {k_ev}")
        spec['harness'] = hc
    if oracle in ('synthetic', 'stub_children'):
        target = synthetic_target if synthetic_target is not None else named_start(lattice, 'plan_2030')
        spec['oracle'].update({'seed': str(synthetic_seed), 'target': list(target), 'alpha': 30000.0, 'f0': 810.0e6,
                               'note': 'SYNTHETIC VALUES -- not evaluations of the system'})
    return spec


def policy_config(spec):
    pol = spec['policy']
    return {'start': list(spec['start']['z']), 'n_max': pol['n_max'], 'arm': spec['arm'], 'R': pol['R'],
            'e_floor': pol['e_floor'], 'block': pol['block'], 'cap': pol['cap'], 'class_rank': pol['class_rank'],
            'domain_size': spec['domain_size'], 'k_ev': pol['k_ev'], 'search_id': spec['search_id']}


def freeze(root, spec):
    if os.path.exists(root) and [n for n in os.listdir(root) if not n.endswith('.log')]:
        raise Refused(f'search root exists and holds files other than launch logs (write-once): {root}')
    os.makedirs(root, exist_ok=True)
    text = P.canonical_json(spec)
    digest = sha256_text(text)
    path = os.path.join(root, f'search_spec_{digest[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    if sha256_file(path) != digest:
        raise Refused('frozen search spec hash mismatch after write')
    return path, digest


def load_spec(root):
    if not os.path.isdir(root):
        raise Refused(f'no search root {root}')
    names = sorted(n for n in os.listdir(root) if n.startswith('search_spec_') and n.endswith('.json'))
    if len(names) != 1:
        raise Refused(f'expected exactly one search_spec_*.json in {root}, found {names}')
    path = os.path.join(root, names[0])
    digest = sha256_file(path)
    if names[0] != f'search_spec_{digest[:8]}.json':
        raise Refused(f'{names[0]} does not carry its own hash prefix ({digest[:8]})')
    spec = load_json(path)
    if spec.get('schema') != SEARCH_SPEC_SCHEMA:
        raise Refused(f'search spec schema {spec.get("schema")!r}')
    return path, digest, spec


def _collect_eval_keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'eval_key' and isinstance(v, str):
                out.add(v)
            else:
                _collect_eval_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_eval_keys(v, out)


def collision_scan(files, keys, own_root, allowed_roots=(), predecessor_roots=()):
    """Pre-run key-collision check over artefacts (campaign / search specs, launch records): every `eval_key` they
    carry is compared with `keys`. Files under `own_root` (the search's own root) and under the declared cache roots
    are EXCLUDED -- a scan that included them would match the artefacts this search itself wrote (CLAUDE.md, the
    W105 rule). `predecessor_roots` (phase spec v2, Planner decision P2): the roots of declared UN-LAUNCHED
    predecessor phases, each verified by the caller (`verify_unlaunched_predecessor`) to hold only its spec and freeze
    logs; excluded and listed separately. Nothing else is excluded.
    -> {'n_files_scanned', 'excluded_own_root', 'excluded_cache_roots', 'excluded_predecessor_roots', 'collisions',
    'refusals'}."""
    own = os.path.realpath(own_root) + os.sep
    allowed = [os.path.realpath(r) + os.sep for r in allowed_roots]
    preds = [os.path.realpath(r) + os.sep for r in predecessor_roots]
    keys = set(keys)
    out = {'n_files_scanned': 0, 'excluded_own_root': [], 'excluded_cache_roots': [], 'excluded_predecessor_roots': [],
           'collisions': [], 'refusals': []}
    for f in files:
        real = os.path.realpath(f)
        if real.startswith(own):
            out['excluded_own_root'].append(f)
            continue
        if any(real.startswith(a) for a in allowed):
            out['excluded_cache_roots'].append(f)
            continue
        if any(real.startswith(a) for a in preds):
            out['excluded_predecessor_roots'].append(f)
            continue
        out['n_files_scanned'] += 1
        found = set()
        try:
            _collect_eval_keys(load_json(f), found)
        except (OSError, ValueError) as error:
            out['refusals'].append(f'collision scan: unreadable {f}: {error}')
            continue
        hit = sorted(found & keys)
        if hit:
            out['collisions'].append({'file': f, 'n_keys': len(hit), 'keys16': [k[:16] for k in hit[:10]]})
    if out['collisions']:
        out['refusals'].append(f"collision scan: {len(out['collisions'])} artefact(s) outside the search's own root "
                               f"and its declared cache roots carry its eval keys: "
                               f"{[c['file'] for c in out['collisions'][:5]]}")
    return out


def freeze_collision_scan(spec, lattice, root, data_root, files=None):
    """At --freeze (harness oracles): every eval key the search can request (the domain x {exit, settled} under the
    spec's harness configuration) against the tracked campaign / search specs and launch records, the search's own
    root, its declared cache roots and (phase spec v2) its declared un-launched predecessor roots -- each RE-VERIFIED
    here as un-launched -- excluded. A collision refuses the freeze. `files`: tests only (default: the tracked
    artefacts, `committed_spec_files`)."""
    oracle = HarnessOracle(spec, lattice, lock_path=None)
    keys = {oracle.eval_key(z, d) for z in enumerate_domain(lattice) for d in P.DEPTHS}
    if len(keys) != 2 * DOMAIN_SIZE:
        raise Refused(f'{len(keys)} distinct eval keys over the domain x 2 depths != {2 * DOMAIN_SIZE}')
    preds = ((spec.get('predecessors') or {}).get('unlaunched') or [])
    for pr in preds:
        verify_unlaunched_predecessor(data_root, pr['root'], pr['spec_sha256'])
    res = collision_scan(committed_spec_files(data_root) if files is None else files, keys, root,
                         allowed_roots=[os.path.join(data_root, c['root']) for c in spec.get('cache_sources') or []],
                         predecessor_roots=[os.path.join(data_root, pr['root']) for pr in preds])
    if res['refusals']:
        raise Refused(res['refusals'])
    out = {'n_keys': len(keys), 'n_files_scanned': res['n_files_scanned'],
           'n_excluded_own_root': len(res['excluded_own_root']),
           'n_excluded_cache_roots': len(res['excluded_cache_roots']), 'collisions': res['collisions'],
           'scope': 'git ls-files data/SRP1/Results: campaign_spec_* / search_spec_* / phase_spec_* / '
                    'launches/launch_*.json'}
    if preds:
        out['n_excluded_predecessor_roots'] = len(res['excluded_predecessor_roots'])
        out['excluded_predecessor_files'] = [os.path.relpath(f, data_root) for f in res['excluded_predecessor_roots']]
    return out


def committed_spec_files(data_root):
    """Tracked campaign / search / phase specs and launch records under data/SRP1/Results (the collision scan's
    scope; phase specs added with the 2x phase build: they carry the cache index's eval keys)."""
    r = subprocess.run(['git', 'ls-files', '--', os.path.join('data', 'SRP1', 'Results')], cwd=data_root,
                       capture_output=True, text=True, check=True)
    names = [n for n in r.stdout.splitlines()
             if os.path.basename(n).startswith(('campaign_spec_', 'search_spec_', 'phase_spec_'))
             or (os.path.basename(n).startswith('launch_') and n.endswith('.json') and '/launches/' in n)]
    return [os.path.join(data_root, n) for n in names]


# ======================================================================================================================
#  the lock (the harness campaign lock file format; explicit stale takeover)
# ======================================================================================================================
def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def acquire_lock(lock_path, search_id, launch_index, root, take_over_stale):
    """O_EXCL create; content = the harness campaign-lock format. An existing lock is taken over only when
    `take_over_stale`, its pid is dead and its campaign id names this search; it is moved to stale_locks/."""
    moved = None
    if os.path.exists(lock_path):
        try:
            content = load_json(lock_path)
        except (OSError, ValueError):
            content = {}
        alive = _pid_alive(content.get('pid'))
        ours = str(content.get('campaign_id', '')).startswith(f'{search_id}_l')
        if alive or not ours or not take_over_stale:
            raise Refused(f'campaign lock held: {lock_path} {content} (pid alive={alive}, names this search={ours}, '
                          f'--take-over-stale-lock={take_over_stale})')
        sdir = os.path.join(root, 'stale_locks')
        os.makedirs(sdir, exist_ok=True)
        m = 1 + len([n for n in os.listdir(sdir) if n.startswith('lock_')])
        moved = os.path.join(sdir, f'lock_{m:03d}.json')
        os.rename(lock_path, moved)
        _log(f'stale lock taken over (pid {content.get("pid")} dead, names this search): moved to {moved}')
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    content = {'pid': os.getpid(), 'campaign_id': f'{search_id}_l{launch_index:03d}', 'campaign_spec_sha256': None,
               'started_utc': _utc(), 'search_root': root}
    os.write(fd, json.dumps(content).encode())
    os.close(fd)
    return content, moved


def set_lock_spec(lock_path, content, spec_sha256):
    content = dict(content, campaign_spec_sha256=spec_sha256)
    tmp = lock_path + '.tmp'
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(content))
    os.replace(tmp, lock_path)
    return content


def release_lock(lock_path):
    if os.path.exists(lock_path) and load_json(lock_path).get('pid') == os.getpid():
        os.remove(lock_path)


# ======================================================================================================================
#  test kill hooks (gate only; test-only specs)
# ======================================================================================================================
class KillHooks:
    def __init__(self, spec, launch_index):
        self.target = None
        raw = os.environ.get(KILL_ENV)
        if raw:
            if not spec.get('test_only'):
                raise Refused(f'{KILL_ENV} is set but the spec is not test-only')
            parts = raw.split(':')
            if parts[0] not in KILL_TRANSITIONS:
                raise Refused(f'{KILL_ENV}: unknown transition {parts[0]}')
            if int(parts[1]) == launch_index:
                self.target = (parts[0], float(parts[2]) if len(parts) > 2 else 0.5)

    def at(self, transition):
        if self.target and self.target[0] == transition:
            _log(f'TEST KILL at {transition}')
            os.killpg(os.getpgid(0), signal.SIGKILL)

    def during_evaluate_start(self):
        """Stub-children oracle: kill the process group (driver + children) `delay` s into `evaluate`."""
        if self.target and self.target[0] == 'during_evaluate':
            delay = self.target[1]

            def _fire():
                time.sleep(delay)
                _log('TEST KILL during evaluate')
                os.killpg(os.getpgid(0), signal.SIGKILL)
            threading.Thread(target=_fire, daemon=True).start()


class _NoHooks:
    def at(self, _t):
        pass

    def during_evaluate_start(self):
        pass


# ======================================================================================================================
#  replay (the resume precondition) and interruption handling
# ======================================================================================================================
def launch_paths(root, n):
    d = os.path.join(root, 'launches')
    return (os.path.join(d, LAUNCH_FMT.format(n)), os.path.join(d, LAUNCH_FMT.format(n) + '.json'),
            os.path.join(d, MANIFEST_FMT.format(n)))


def _complete_indices(root):
    d = os.path.join(root, 'launches')
    if not os.path.isdir(d):
        return [], []
    names = os.listdir(d)
    idx = sorted({int(n[7:10]) for n in names if n.startswith('launch_') and n[7:10].isdigit()})
    complete = [n for n in idx if all(os.path.exists(p) for p in launch_paths(root, n)[1:])]
    return idx, complete


def handle_interruptions(root, spec, oracle_kind, results_files=None):
    """Move every incomplete launch (and results without their manifest) aside, recording the killed evaluations
    with their hours. -> the interruption record (or None). `results_files` = (results, its manifest); default the
    search's (the phase passes its own)."""
    idx, complete = _complete_indices(root)
    if complete != list(range(1, len(complete) + 1)):
        raise Refused(f'complete launches {complete} are not contiguous 1..n')
    incomplete = [n for n in idx if n not in complete]
    if incomplete and incomplete != [len(complete) + 1]:
        raise Refused(f'incomplete launches {incomplete}: only launch {len(complete) + 1} may be incomplete')
    paths = []
    for n in incomplete:
        paths += [p for p in launch_paths(root, n) if os.path.exists(p)]
    rf, rm = results_files or (RESULTS_FILE, RESULTS_MANIFEST)
    res, man = os.path.join(root, rf), os.path.join(root, rm)
    if os.path.exists(res) and not os.path.exists(man):
        paths.append(res)
    if not paths:
        return None
    idir = os.path.join(root, 'interrupted')
    os.makedirs(idir, exist_ok=True)
    m = 1 + len([n for n in os.listdir(idir) if n.startswith('interrupted_') and n.endswith('.json')])
    aside = os.path.join(idir, f'aside_{m:03d}')
    os.makedirs(aside)
    killed = []
    for n in incomplete:
        ldir = launch_paths(root, n)[0]
        evals = os.path.join(ldir, 'evals')
        for name in sorted(os.listdir(evals)) if os.path.isdir(evals) else []:
            if name in IGNORED_FILE_NAMES:
                continue
            ed = os.path.join(evals, name)
            lj = os.path.join(ed, 'launch.json')
            started = None
            if os.path.isfile(lj):
                try:
                    j = load_json(lj)
                    started = j.get('started_t')
                    if started is None and j.get('started_utc'):
                        started = datetime.fromisoformat(j['started_utc']).timestamp()
                except (OSError, ValueError):
                    started = None
            mt = [os.path.getmtime(f) for f in files_under(ed)]
            last = max(mt) if mt else None
            killed.append({'launch': n, 'eval_dir_name': name,
                           'record_present': os.path.isfile(os.path.join(ed, 'evaluation_record.json')),
                           'started_t': started, 'last_write_t': last,
                           'hours_lower_bound': ((last - started) / 3600.0) if (started and last) else None,
                           'counted': False})
    moved = []
    for p in paths:
        dest = os.path.join(aside, os.path.basename(p))
        os.rename(p, dest)
        moved.append({'from': os.path.relpath(p, root), 'to': os.path.relpath(dest, root)})
    rec = {'schema': 'p515_s56_search_interrupted_v1', 'index': m, 'detected_utc': _utc(), 'detected_by_pid': os.getpid(),
           'incomplete_launches': incomplete, 'moved_aside': moved, 'killed_evaluations': killed,
           'oracle': oracle_kind, 'rule': 'an interrupted launch never counts; nothing deleted; the action re-issues',
           'hours_killed_lower_bound_total': sum(k['hours_lower_bound'] or 0.0 for k in killed)}
    write_once_json(os.path.join(idir, f'interrupted_{m:03d}.json'), rec)
    _log(f'interruption {m}: moved aside {len(moved)} path(s); {len(killed)} killed evaluation(s), uncounted')
    return rec


def replay(root, spec, spec_sha, lattice, env, oracle, cache):
    """Re-derive the history from the launch chain (module docstring). -> (state, records, chain).
    Raises Refused on any disagreement."""
    _idx, complete = _complete_indices(root)
    state = P.initial_state(policy_config(spec))
    records = dict(cache)
    chain, prev_manifest, states = [], None, []
    for n in complete:
        ldir, rec_p, man_p = launch_paths(root, n)
        lr, manifest = load_json(rec_p), load_json(man_p)
        bad = []
        if lr.get('schema') != LAUNCH_RECORD_SCHEMA or lr.get('launch_index') != n:
            bad.append('schema / index')
        if lr.get('search_spec_sha256') != spec_sha:
            bad.append('search spec sha256')
        if lr.get('previous_launch_manifest_sha256') != prev_manifest:
            bad.append('hash chain')
        rel_rec = os.path.relpath(rec_p, root)
        if manifest.get(rel_rec) != sha256_file(rec_p):
            bad.append('manifest does not hold the launch record hash')
        expected = {os.path.relpath(f, root) for f in files_under(ldir)} | {rel_rec}
        if set(manifest) != expected:
            bad.append(f'manifest file set differs ({len(set(manifest) ^ expected)} paths)')
        if any(sha256_file(os.path.join(root, f)) != h for f, h in manifest.items() if os.path.isfile(
                os.path.join(root, f))):
            bad.append('a manifest hash does not hold')
        state, action = P.step(state, records, env)
        if P.canonical_json(action) != P.canonical_json(lr.get('action')):
            bad.append(f'replayed action {action} != recorded {lr.get("action")}')
        if P.canonical_json(state) != P.canonical_json(lr.get('state_with_action')):
            bad.append('replayed state != recorded state_with_action')
        states.append(state)
        if action.get('kind') == 'evaluate':
            bad += oracle.verify_launch(ldir, action)
        got = []
        for r in lr.get('records') or []:
            z, depth = tuple(r['z']), r['depth']
            key = oracle.eval_key(z, depth)
            ed = os.path.join(root, r['eval_dir'])
            if key != r['eval_key'] or not os.path.basename(ed).startswith(key[:16]):
                bad.append(f'eval key of {list(z)} {depth} does not recompute')
                continue
            if os.path.dirname(os.path.dirname(ed)) != ldir:
                bad.append(f'eval dir {r["eval_dir"]} is not inside launch {n}')
                continue
            summ, rfile = oracle.summary_from_dir(ed, z, depth)
            if sha256_file(rfile) != r['record_sha256'] or P.canonical_json(summ) != P.canonical_json(r['summary']):
                bad.append(f'record of {list(z)} {depth} does not re-derive from its eval dir')
            got.append(((z, depth), summ))
        if action.get('kind') != 'evaluate' or sorted(P.canonical_json([list(k[0]), k[1]]) for k, _ in got) != \
                sorted(P.canonical_json([z, action['depth']]) for z in action['zs']):
            bad.append('the launch records are not exactly its action')
        if bad:
            raise Refused([f'launch {n}: {b}' for b in bad])
        for k, s in got:
            if k in records:
                raise Refused(f'launch {n}: record {k} already present (write-once)')
            records[k] = s
        chain.append({'n': n, 'manifest_sha256': sha256_file(man_p), 'record_sha256': manifest[rel_rec]})
        prev_manifest = chain[-1]['manifest_sha256']
    allowed_top = {os.path.basename(load_spec(root)[0]), 'launches', 'interrupted', 'stale_locks', STATE_VIEW,
                   RESULTS_FILE, RESULTS_MANIFEST}
    foreign = [n for n in os.listdir(root) if n not in allowed_top and not n.endswith('.log')]
    if foreign:
        raise Refused(f'foreign files in the search root: {foreign}')
    return state, records, chain, states


# ======================================================================================================================
#  one launch
# ======================================================================================================================
def _results_payload(spec, spec_sha, state, records, chain):
    return {'schema': RESULTS_SCHEMA, 'search_id': spec['search_id'], 'search_spec_sha256': spec_sha,
            'arm': spec['arm'], 'oracle': spec['oracle'], 'n_launches': len(chain),
            'last_launch_manifest_sha256': chain[-1]['manifest_sha256'] if chain else None,
            'terminal_report': P.terminal_report(state, records), 'state': state,
            'objective_convention': ('F = I(x) + Q(x) as the evaluation summary carries it; SYNTHETIC values for '
                                     'synthetic / stub-children oracles' if spec['oracle']['kind'] != 'harness'
                                     else 'F = I(x) + gross_operational_cost - TV credit (oracle mapping '
                                          f'{MAPPING_SCHEMA}), EUR of total discounted planning cost')}


def run_launch(root, *, data_root, lock_path, take_over_stale, cache=None):
    """ONE launch (module docstring). -> exit code."""
    spec_path, spec_sha, spec = load_spec(root)
    lattice, _inst = build_lattice(data_root)
    env = LatticeEnv(lattice)
    oracle = make_oracle(spec, lattice, lock_path)
    _idx, complete = _complete_indices(root)
    n_next = len(complete) + 1
    hooks = KillHooks(spec, n_next)
    lock, _moved = acquire_lock(lock_path, spec['search_id'], n_next, root, take_over_stale)
    if isinstance(oracle, HarnessOracle):
        oracle.lock_content = lock
    try:
        hooks.at('after_lock')
        if cache is None and spec.get('cache_sources') and isinstance(oracle, HarnessOracle):
            cache = load_cache(spec, oracle, data_root)   # the frozen cache index, re-verified and mapped
        handle_interruptions(root, spec, oracle.kind)
        state, records, chain, states = replay(root, spec, spec_sha, lattice, env, oracle, cache or {})
        state, action = P.step(state, records, env)
        view_p = os.path.join(root, STATE_VIEW)
        if os.path.exists(view_p):  # the derived view must equal the replayed state of the launch it names
            view = load_json(view_p)
            v = view.get('n_launches')
            if not isinstance(v, int) or not 0 <= v <= len(chain):
                raise Refused(f'the state view names launch {v}; the chain holds {len(chain)}')
            replayed = state if v == len(chain) else states[v]
            if P.canonical_json(view.get('state')) != P.canonical_json(replayed):
                raise Refused(f'the state view (launch {v}) does not equal the replayed state')
        if action['kind'] == 'terminated':
            atomic_json(view_p, {'n_launches': len(chain), 'state': state, 'state_sha256':
                                 sha256_text(P.canonical_json(state)), 'derived': 'replayed from the launch chain'})
            if not os.path.exists(os.path.join(root, RESULTS_MANIFEST)):
                _write_results(root, spec, spec_sha, state, records, chain, hooks)
            return EXIT_REVIEW if state['termination']['reason'] in REVIEW_TERMINATIONS else EXIT_TERMINATED
        ldir, rec_p, man_p = launch_paths(root, n_next)
        preflight = None
        if isinstance(oracle, HarnessOracle) and not oracle.stub:  # real launch: preconditions + memory, zero solves
            preflight = real_launch_preflight(oracle.H, ldir, min(int(spec['harness']['concurrency']),
                                                                  len(action['zs'])),
                                              spec['harness']['memory_per_slot_gib'], data_root,
                                              mapping=spec.get('mapping'))
        started, started_utc = time.time(), _utc()
        os.makedirs(ldir)
        hooks.at('after_launch_dir')
        produced = oracle.evaluate(action, ldir, hooks, launch_index=n_next)
        hooks.at('after_evaluate')
        recs = []
        for r in produced:
            summ, rfile = oracle.summary_from_dir(r['eval_dir'], r['z'], r['depth'])
            recs.append({'z': r['z'], 'depth': r['depth'], 'eval_key': r['eval_key'],
                         'eval_dir': os.path.relpath(r['eval_dir'], root), 'record_file': os.path.relpath(rfile, root),
                         'record_sha256': sha256_file(rfile), 'summary': summ})
        lr = {'schema': LAUNCH_RECORD_SCHEMA, 'launch_index': n_next, 'search_id': spec['search_id'],
              'search_spec_sha256': spec_sha, 'previous_launch_manifest_sha256': chain[-1]['manifest_sha256']
              if chain else None, 'action': action, 'state_with_action': state, 'records': recs,
              'started_utc': started_utc, 'finished_utc': _utc(), 'wall_s': time.time() - started,
              'pid': os.getpid(), 'driver_sha256': sha256_file(os.path.abspath(__file__)),
              'policy_sha256': sha256_file(P.__file__), 'oracle': oracle.kind,
              **({'real_launch_preflight': preflight} if preflight is not None else {})}
        rec_sha = write_once_json(rec_p, lr)
        hooks.at('after_launch_record')
        manifest = {os.path.relpath(f, root): sha256_file(f) for f in files_under(ldir)}
        manifest[os.path.relpath(rec_p, root)] = rec_sha
        man_sha = write_once_json(man_p, manifest)
        hooks.at('after_manifest')
        chain.append({'n': n_next, 'manifest_sha256': man_sha, 'record_sha256': rec_sha})
        for r in recs:
            records[(tuple(r['z']), r['depth'])] = r['summary']
        state2, action2 = P.step(state, records, env)
        atomic_json(view_p, {'n_launches': len(chain), 'state': state2,
                             'state_sha256': sha256_text(P.canonical_json(state2)),
                             'derived': 'replayed from the launch chain'})
        hooks.at('after_state_view')
        _log(f'launch {n_next}: {action["purpose"]} {action["depth"]} x{len(action["zs"])}; budget '
             f'{state2["budget_used"]}/{state2["config"]["n_max"]}; next: {action2["kind"]}')
        if action2['kind'] == 'terminated':
            _write_results(root, spec, spec_sha, state2, records, chain, hooks)
            return EXIT_REVIEW if state2['termination']['reason'] in REVIEW_TERMINATIONS else EXIT_TERMINATED
        return EXIT_SUSPENDED
    finally:
        release_lock(lock_path)


def _write_results(root, spec, spec_sha, state, records, chain, hooks):
    res = os.path.join(root, RESULTS_FILE)
    sha = write_once_json(res, _results_payload(spec, spec_sha, state, records, chain))
    hooks.at('after_results')
    write_once_json(os.path.join(root, RESULTS_MANIFEST), {RESULTS_FILE: sha})
    _log(f'terminated: {state["termination"]["reason"]}; results written')


# ======================================================================================================================
#  dry run
# ======================================================================================================================
def dry_run(out_dir, data_root, starts=('x0', 'reference_unit', 'plan_2030'), seed='dry-run-1', n_max=120):
    """Zero solves: for each named start, the first poll order and the whole search against the synthetic oracle
    (in process, no files but `out_dir`)."""
    lattice, inst = build_lattice(data_root)
    env = LatticeEnv(lattice)
    out = {'schema': 'p515_s56_search_dry_run_v1', 'utc': _utc(), 'instance': inst, 'policy_sha256':
           sha256_file(P.__file__), 'driver_sha256': sha256_file(os.path.abspath(__file__)), 'seed': seed,
           'n_max': n_max, 'note': 'SYNTHETIC VALUES -- not evaluations of the system', 'starts': {}}
    for name in starts:
        z0 = named_start(lattice, name)
        first = P.order(lattice, z0, None, P.CLASS_RANK, env.neighbourhood_points(z0))
        land = SyntheticLandscape(seed, named_start(lattice, 'plan_2030'), z0)
        state = P.initial_state({'start': list(z0), 'n_max': n_max, 'arm': 'dry_run', 'domain_size': DOMAIN_SIZE})
        actions = []

        def ev(action, land=land, actions=actions):
            actions.append({'purpose': action['purpose'], 'depth': action['depth'], 'n': len(action['zs'])})
            return {(tuple(z), action['depth']): land.summary(z, action['depth']) for z in action['zs']}

        state, records = P.run_to_end(state, {}, env, ev)
        out['starts'][name] = {'z': list(z0), 'label': lattice.label(z0), 'N': len(first),
                               'first_poll_order': [{'pos': r['pos'], 'label': r['label'], 'class': r['class'],
                                                     'move': r['move']} for r in first],
                               'actions': actions, 'terminal_report': P.terminal_report(state, records),
                               'incumbent_history': state['incumbent_history'],
                               'state_sha256': sha256_text(P.canonical_json(state))}
        _log(f'dry run {name}: |N| {len(first)}; {len(actions)} actions; '
             f'{state["termination"]["reason"]}; budget {state["budget_used"]}')
    os.makedirs(out_dir)
    sha = write_once_json(os.path.join(out_dir, 'search_dry_run.json'), out)
    write_once_json(os.path.join(out_dir, 'search_dry_run_manifest_sha256.json'), {'search_dry_run.json': sha})
    return out


# ======================================================================================================================
#  THE 2x PHASE (TASKS.md "Author decisions on report (i)", 2026-11-24, step 5a). Decisions: `P.phase_step` (pure).
#  This section: the phase spec (freeze), the cache (pilot records + their exit views), one launch = one block, the
#  replay, the 120 h ceiling, the preflight-only mode and the results. Built in this module (not a new file) so that
#  the oracles, the lock, the interruption handling and the collision scan are the search's own code (Worker's call,
#  logged in the report).
# ======================================================================================================================
#  PHASE ROOT (write-once; nothing deleted)
#    phase_spec_<sha8>.json        frozen by --phase-freeze
#    launches/launch_<NNN>/ (+ .json, _manifest_sha256.json)   ONE launch = ONE block of <= 5 evaluations at one depth
#    interrupted/, stale_locks/    as the search
#    ceiling/ceiling_stop_<MMM>.json   a launch refused by the 120 h ceiling (hours summary; nothing evaluated)
#    phase_state.json              derived view (replayed)
#    phase_results.json (+ manifest)   written once, when the phase terminates
#    launch_<NNN>_stdout.log / _stderr.log   the author's loop (NNN = its invocation counter)
#  EXIT CODES: 4 a block ran, more work remains; 0 the phase completed; 5 review (harness_error, offset above e,
#  not_settled review trigger, start not certified -- the reason is printed and in phase_state.json); 6 the 120 h
#  ceiling (hours summary written, nothing launched); 2 refused (nothing written); 1 crash.
#  CEILING (author, 2026-11-24): before each block, cumulative measured machine hours (the sum of the completed phase
#  launches' walls, plus the wall of every interrupted launch from its killed evaluations' first start to last write)
#  + the projected wall of the next block must not exceed ceiling_h = 120. Projected wall of a block at depth d =
#  max(the frozen projection for d, the largest measured wall of a completed phase launch at d). Frozen projections
#  (from the pilot's committed records): exit block = kappa psi t_exit; settled block = kappa psi t_300 (t_300 = mean
#  child wall to K = 300 of the 7 pilot runs).
#  CACHE: the declared cache roots (the pilot root) are indexed by exact eval key (`build_cache_index`): x0_i1,
#  unit_i1 and staged_i1 at the settled key (K 300, stop-disabled, pricing capture, rho0 x 1). The EXIT summary of a
#  cached settled record is its EXIT VIEW (`map_record(files, 'exit')`: read at the would-be certification cycle --
#  the reading the pilot calibrated e_EV on), served only when no exit-depth record of that z exists and tagged
#  `served_as` = 'exit_view_of_settled_cache_record' with the settled key and dir. Exit views are served only from
#  a cache source whose `exit_views` is True (spec v2: the pilot source; False on a continuation predecessor, and on
#  the pilot source of a continuation frozen with --no-pilot-exit-views, the R4 fallback).
#  SPEC v2 (Advisor pre-launch review GO-WITH-CHANGES; Planner decisions R1, R4, P1, P2, TASKS 5a, 2026-11-24):
#    - efficiency measure P1 and the exit-view check R4: `P.phase_step` (policy section comment);
#    - PREDECESSORS: `predecessors.unlaunched` = declared predecessor phase roots verified UN-LAUNCHED (only the spec
#      and the freeze logs; `verify_unlaunched_predecessor`), excluded from the collision scan and re-verified at every
#      launch; `predecessors.continuation` = declared predecessor phase roots WITH launches, stopped at exit 5 or 6,
#      indexed as cache roots from their complete launch records (`continuation_source`), their measured hours
#      carried into the ceiling (`phase.ceiling.carried_h`), their launch count re-verified at every launch;
#    - CODE DRIFT: every --phase-run (preflight-only included) recomputes `code_hashes()` and refuses (exit 2), naming
#      the file, if any differs from the spec's `code_sha256`;
#    - files named exactly .DS_Store are ignored wherever the driver counts or manifests files (IGNORED_FILE_NAMES).
PHASE_SPEC_SCHEMA = 'p515_s56_phase_spec_v2'
PHASE_LAUNCH_SCHEMA = 'p515_s56_phase_launch_record_v1'
PHASE_RESULTS_SCHEMA = 'p515_s56_phase_results_v1'
PHASE_RESULTS_FILE, PHASE_RESULTS_MANIFEST = 'phase_results.json', 'phase_results_manifest_sha256.json'
PHASE_STATE_VIEW = 'phase_state.json'
EXIT_CEILING = 6
PHASE_PILOT_ROOT_REL = os.path.join(RESULTS_PARENT_REL, 'pilot_ev_v1')
PHASE_PILOT_RESULT_REL = os.path.join(RESULTS_PARENT_REL, 'pilot_ev_v1_result_r2', 'pilot_result.json')
PHASE_K_EV = 300                    # Planner decision on the pilot (TASKS 2026-11-24)
PHASE_E_EV_ROUNDED = 17494          # e_EV = 17,494 EUR (Planner decision; the frozen pilot rule's value is read)
PHASE_MEMORY_PER_SLOT_GIB = 3.29    # Planner decision: measured peak x 1.15
PHASE_CEILING_H = 120.0             # author decision 1 (2026-11-24)
PHASE_FLEX = 2.0
#  THE 1x PHASE (TASKS.md step 5d; author, 2026-11-26: 1x phase authorized, ceiling 60 h, same execution, budget kept):
#  the 2x phase's design and configuration with the declared price level `flex_price` = 1.0 (--phase-flex) and NOTHING
#  else changed (R, K_EV, e_EV, the efficiency rule, the screen / ladder, the search policy, the gates, the mapping).
#  Planner decisions (TASKS 5d, 2026-11-26): (D1) e_EV and K_EV carry over from the 2x pilot (no 1x pilot); the offset
#  review stop and the exit-view check (Q7) guard the carry-over. (D2) no record carries over -- the price level is in
#  the eval key -- so x0, the reference unit and the staged plan are evaluated by the phase itself: the R4 path
#  (pilot_exit_views False: x0 gets an exit-depth run; x0's settled run in the candidate confirmation; the unit's and
#  the staged plan's settled runs in the pool), with NO cache source (the pilot root is not declared: it carries no 1x
#  key, and undeclared it stays inside the collision scan). The exit-view check keeps mode 'stop' (Q7 "as at 2x"):
#  'record' is kept for a continuation that serves no exit view. A 1x spec carries `phase.price_level`; a 2x spec is
#  built exactly as before (no new field). Default 2.0: every 2x call is unchanged.
PHASE_FLEX_LEVELS = (2.0, 1.0)
PHASE_CEILING_1X_H = 60.0           # author, 2026-11-26 (TASKS 5d)
PHASE_PREDICTION_IDS_1X = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9')
PHASE_AUTHORITY_1X = ['TASKS.md step 5d (2026-11-26): author decisions on the 2x report -- 2x controls recorded as '
                      'decided at zero solves; 1x phase authorized, ceiling 60 h, execution as at 2x (author\'s '
                      'Terminal, app closed, 5 at a time, first-block hold then loop); budget B = 1,000,000 EUR kept',
                      'TASKS.md step 5d: Planner decisions D1 (e_EV = 17,494 EUR and K_EV = 300 carry over from the 2x '
                      'pilot; no 1x pilot) and D2 (no record carries over: the price level is in the eval key; x0, the '
                      'reference unit and the staged plan evaluated by the phase itself on the R4 path; no cache source)',
                      'P5_15_S56_PHASE2X_REPORT.md (50d7dc67) sections 7b and 8',
                      'configuration: phase2x_ev_v3/phase_spec_600bbceb.json with flex_price_multiplier 1.0; mapping '
                      'v2 extended solve-count reconciliation (4030508f); identity tolerance (28ce9f08)',
                      'TASKS.md step 5a: Advisor pre-launch review GO-WITH-CHANGES; Planner decisions R1, R4, R5, P1, P2 '
                      '(2026-11-24): phase spec v2',
                      'S56_SEARCH_DRIVER_DESIGN.md (all sections)', 'S56_PLAN_v4_1.md sections 3-4']
PHASE_PROJECTION_1X = {
    'source': 'P5_15_S56_PHASE2X_REPORT.md (50d7dc67) section 7b, restated by the Planner (TASKS 5d): a PROJECTION, '
              'not a prediction',
    'two_candidates_no_search_move_h': [16.7, 32.8],
    'no_confirm_candidate_in_the_screen_h': [7.8, 22.0],
    'each_search_move_adds_h': [8.0, 22.0],
    'note': 'the spec field projected_hours is computed as at 2x and assumes x0, the unit and the staged plan cached '
            '(they are not at 1x: + 1 exit launch for x0 and + 1 settled launch for the unit and the staged plan, '
            'which the restated projection includes)'}


def phase_flex_level(flex_price):
    """The declared price level of a phase (allowed: PHASE_FLEX_LEVELS) -> float; raises Refused otherwise."""
    if isinstance(flex_price, bool) or not isinstance(flex_price, (int, float)) or \
            float(flex_price) not in PHASE_FLEX_LEVELS:
        raise Refused(f'phase price level {flex_price!r} not in {PHASE_FLEX_LEVELS}')
    return float(flex_price)


def phase_flex_of(spec):
    """The price level a (frozen) phase spec runs at: its harness entry options' flex_price_multiplier, equal at both
    depths (a 2x spec carries 2.0 there; a 1x spec 1.0, and `phase.price_level` beside it). None for a synthetic spec
    (no harness configuration)."""
    hz = spec.get('harness')
    if not hz:
        return None
    lv = {hz['by_depth'][d]['entry_options'].get('flex_price_multiplier') for d in P.DEPTHS}
    if len(lv) != 1:
        raise Refused(f'phase spec {spec.get("phase_id")}: price levels differ across depths {sorted(map(str, lv))}')
    return lv.pop()


PHASE_EXIT_CAP = 300                # Worker reading, logged: the exit depth runs to the same cap as the pilot runs
PHASE_CONCURRENCY = 5               # author decision 2: 5 at a time
PHASE_N_CANDIDATES = 2              # author decision 3
PHASE_LADDER_ZP = (2, 4)            # 0.5 and 1.0 MVA (Planner decision: size ladder)
PHASE_LADDER_N_NODE_YEARS = 2
PHASE_SCREEN_PAIRS = ((1, 2), (1, 1))   # 0.25 MVA with 1 MWh, 0.25 MVA with 0.5 MWh
PHASE_PREDICTION_IDS = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7')   # Q7: R4 (spec v2)
PHASE_AUTHORITY = ['TASKS.md: Author decisions on report (i) (2026-11-24); step 5a',
                   'TASKS.md step 5a: Advisor pre-launch review GO-WITH-CHANGES; Planner decisions R1, R4, R5, P1, P2 '
                   '(2026-11-24): phase spec v2',
                   'P5_15_S56_REPORT_i.md; P5_15_S56_PILOT_REPORT.md', 'S56_SEARCH_DRIVER_DESIGN.md (all sections)',
                   'S56_PLAN_v4_1.md sections 3-4']


def phase_harness_config(data_root, flex_price=PHASE_FLEX):
    """The phase's per-depth harness configuration, taken from the pilot spec's x0_i1 entry (rho0 x 1): settled =
    its options (stop-disabled + pricing capture) at cap K_EV = 300; exit = the same without those two keys, at cap
    PHASE_EXIT_CAP. The staged plan is per entry. 1x phase (TASKS 5d): the same, with flex_price_multiplier set to the
    declared price level 1.0 (nothing else changed); 2.0 (default) returns exactly what it always did."""
    import p515_s44_campaign_harness as H
    flex = phase_flex_level(flex_price)
    sp, sha, ps = load_pilot_spec(os.path.join(data_root, PHASE_PILOT_ROOT_REL))
    e = next(x for x in ps['entries'] if x['label'] == 'x0_i1')
    opts = copy.deepcopy(e['options'])
    opts.pop(H.S56_STAGED_INVESTMENT_KEY)
    if opts.get(H.S56_SETTLED_STOP_DISABLED_KEY) is not True or opts.get(H.S56_PRICING_CAPTURE_KEY) is not True or \
            'rho0_scale' in opts or opts.get('flex_price_multiplier') != PHASE_FLEX or ps['k_pilot'] != PHASE_K_EV:
        raise Refused('the pilot x0_i1 entry is not the settled 2x rho0 x 1 configuration at K 300')
    if flex != PHASE_FLEX:   # the 1x phase: the pilot's configuration at the declared price level, nothing else
        opts['flex_price_multiplier'] = flex
    exit_opts = {k: v for k, v in opts.items() if k not in (H.S56_SETTLED_STOP_DISABLED_KEY, H.S56_PRICING_CAPTURE_KEY)}
    return {'by_depth': {'exit': {'configuration': copy.deepcopy(ps['configuration']), 'cap': PHASE_EXIT_CAP,
                                  'entry_options': exit_opts},
                         'settled': {'configuration': copy.deepcopy(ps['configuration']), 'cap': PHASE_K_EV,
                                     'entry_options': opts}},
            'concurrency': PHASE_CONCURRENCY, 'required_consecutive_cycles': ps['required_consecutive_cycles'],
            'memory_per_slot_gib': PHASE_MEMORY_PER_SLOT_GIB,
            'source': dict({'pilot_spec': os.path.relpath(sp, data_root), 'pilot_spec_sha256': sha,
                            'entry': 'x0_i1 options without the staged plan; exit depth also without the stop-disabled '
                                     'and pricing-capture keys'},
                           **({} if flex == PHASE_FLEX else {
                               'price_level': f'flex_price_multiplier set to {flex} (the pilot ran at {PHASE_FLEX}); '
                                              f'nothing else changed (TASKS 5d)'}))}


def phase_projection_inputs(data_root):
    """The frozen block projections, from the pilot's committed records (zero solves)."""
    res = load_json(os.path.join(data_root, PHASE_PILOT_RESULT_REL))
    root = os.path.join(data_root, PHASE_PILOT_ROOT_REL)
    walls, launch_walls = [], []
    for n in _complete_indices(root)[1]:
        lr = load_json(launch_paths(root, n)[1])
        launch_walls.append(lr['wall_s'] / 3600.0)
        for r in lr['records']:
            rec = load_json(os.path.join(root, r['eval_dir'], 'evaluation_record.json'))
            walls.append(rec['wall_time_s']['child_process_s'] / 3600.0)
    t300 = sum(walls) / len(walls)
    psi, t_exit = res['psi'], res['t_exit_h']
    return {'t_exit_h': t_exit, 'psi': psi, 'kappa': KAPPA_CONTENTION, 't_300_h': t300,
            'child_walls_to_300_h': walls, 'pilot_launch_walls_h': launch_walls, 'e_EV': res['e_EV'],
            'K_EV_frozen_rule': res['K_EV'],
            'projection_h': {P.DEPTH_EXIT: KAPPA_CONTENTION * psi * t_exit, P.DEPTH_SETTLED: KAPPA_CONTENTION * psi * t300},
            'formula': 'exit block = kappa psi t_exit; settled block = kappa psi t_300 (t_300 = mean pilot child wall '
                       'to K = 300); the design count formula per block, kappa and psi as measured',
            'source': {'pilot_result': PHASE_PILOT_RESULT_REL,
                       'pilot_result_sha256': sha256_file(os.path.join(data_root, PHASE_PILOT_RESULT_REL)),
                       'pilot_root': PHASE_PILOT_ROOT_REL}}


def validate_phase_predictions(pred, ids=PHASE_PREDICTION_IDS):
    if not isinstance(pred, dict) or not isinstance(pred.get('predictions'), dict):
        raise Refused('the predictions file must hold {"predictions": {...}}')
    missing = [k for k in ids if k not in pred['predictions']]
    bad = [k for k, v in pred['predictions'].items()
           if not all(isinstance(v.get(f), str) and v[f] for f in ('statement', 'evaluated_as', 'on_failure'))]
    if missing or bad:
        raise Refused(f'phase predictions incomplete: missing {missing}; without statement / evaluated_as / '
                      f'on_failure {bad}')
    return pred


def phase_capture_checklist(H, price_level_1x=False):
    """Rule eleven for the phase, before any run: a capture path for every quantity the phase rules and the recorded
    predictions read. -> checklist; raises Refused naming the failed checks. `price_level_1x` (the 1x phase, TASKS
    5d): also the paths its added predictions read (Q8 price level, Q9 leak) and the ESSO recovery reconciliation
    (4030508f); the 2x checklist is unchanged."""
    import inspect
    rec_src = inspect.getsource(H.build_evaluation_record)
    run_src = inspect.getsource(phase_run)
    checks = {
        'mapping capture paths (F_exit, bar, settled F, V, gates; Q1, Q3)': bool(assert_mapping_capture_paths(H)),
        'local_solve_failures in the record (Q2)': "'local_solve_failures'" in rec_src,
        'harness_error from the mapping (Q2)': "status='harness_error'" in inspect.getsource(map_record),
        'peak_rss in the record (Q5)': "'peak_rss'" in rec_src and 'child_python_process_ru_maxrss' in
                                       inspect.getsource(H),
        'launch wall in the phase launch record (Q4, ceiling)': "'accounted_wall_s'" in run_src,
        'termination in the phase state (Q4)': "'termination'" in inspect.getsource(P._phase_terminate),
        'efficiency record in the phase state (Q6)': 'efficiency_rule(' in inspect.getsource(P.phase_step),
        'efficiency per-class counts and the v1 band measure (Q6, P1)': "'class_counts'" in inspect.getsource(
            P.efficiency_rule) and "'band_measure_information_only'" in inspect.getsource(P.efficiency_rule),
        'exit_view in the settled summary (Q7, R4)': "exit_view={k: exit_view.get(k)" in inspect.getsource(map_record)
                                                     and "'read_cycle'" in inspect.getsource(map_record),
        'exit-view check per candidate in the phase state (Q7, R4)': "c['exit_view_check']" in inspect.getsource(
            P.phase_step),
        'best start in the phase state (Q3)': "'chosen'" in inspect.getsource(P.phase_step),
        'per-cycle trajectory by default (stage template)': {'recourse', 'cycle_convergence', 'cycle'}
                                                            <= set(H.PER_CYCLE_TRAJECTORY_FIELDS),
    }
    if price_level_1x:
        map_src = inspect.getsource(map_record)
        child_src = inspect.getsource(H._child_real)
        checks.update({
            'leak priced as conversion loss in every settled summary (Q9)':
                "leak=pc['leak']" in map_src and "'priced_loss_eur': g_leak" in inspect.getsource(pricing_components)
                and "'priced_loss_eur'" in inspect.getsource(H.s56_leak_detector),
            'leak per settled evaluation in the phase prediction inputs (Q9)':
                "'Q9'" in inspect.getsource(phase_prediction_inputs),
            'ESSO recovery reconciliation in the guards (Q2; 4030508f)':
                "solve_profile_reconciliation(record, files)['holds']" in inspect.getsource(guards_exact)
                and "'esso_recovery_solves'" in inspect.getsource(solve_profile_reconciliation)
                and "'esso_recovery_events'" in inspect.getsource(read_eval_files),
            'price level applied and read back in the child, recorded (Q8)':
                "holder['flex_price_readback_terminal']" in child_src
                and "'flex_price_readback_terminal'" in child_src
                and "'flex_price_multiplier'" in inspect.getsource(H._flex_price_record_fields),
            'screen dF_exit per row in the phase results (Q1, Q8)':
                "'dF_exit': cls['dF']" in inspect.getsource(P._table_row),
        })
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise Refused(f'phase capture path missing: {failed}')
    return checks


def _screen_and_ladder_tables(lattice):
    """Every screen plan and every possible ladder point, checked against the lattice (C5 rules: `reasons`) and
    canonicalised with GeneralLattice (canonicalize of the node-year form == canonical_of_z; z_of_canonical
    round-trips; key)."""
    import p516_lattice as GL

    def row(n, y, zp, ze):
        z = lattice.with_pair(lattice.x0(), n, y, zp, ze)
        canon = lattice.canonicalize({str(n): {str(y): [GL.GRANULE_P_MVA * zp, GL.GRANULE_E_MWH * ze]}})
        return {'node': n, 'year': y, 'zp': zp, 'ze': ze, 'z': list(z), 'label': lattice.label(z),
                'p_mva': GL.GRANULE_P_MVA * zp, 'e_mwh': GL.GRANULE_E_MWH * ze,
                'reasons': lattice.reasons(z), 'investment_cost_eur': lattice.investment_cost(z),
                'canonical_matches': canon == lattice.canonical_of_z(z),
                'round_trip': lattice.z_of_canonical(canon) == tuple(z), 'key': lattice.key(z)}
    screen = [row(n, y, zp, ze) for n in lattice.nodes for y in lattice.years for zp, ze in PHASE_SCREEN_PAIRS]
    bad = [r['label'] for r in screen if r['reasons'] or not r['canonical_matches'] or not r['round_trip']]
    if bad or len(screen) != 18:
        raise Refused(f'screen plans refused by the lattice / canonical form: {bad} (n = {len(screen)})')
    ladder = [dict(row(n, y, zp, P.DURATION_RATIO[d] * zp), duration=d) for n in lattice.nodes for y in lattice.years
              for d in P.DURATION_RATIO for zp in PHASE_LADDER_ZP]
    if any(not r['canonical_matches'] or not r['round_trip'] for r in ladder):
        raise Refused('a ladder point does not canonicalise')
    return screen, ladder


PROJECTION_FINAL_POLL_CONFIRM_SHARES = ((0, 1), (1, 4), (1, 2))


def phase_hours_projection(lattice, screen, pool_z, projection_h, block=P.BLOCK):
    """PROJECTED phase hours (zero solves; the design's count with the frozen per-block projections). Cache at the
    candidate polls = the screen plans + the pool members (ladder plans ignored: unknown before the screen); the two
    candidates are bracketed by the two smallest and the two largest uncached |N| over the screen plans. Search from
    the pilot staged plan (the pool's lowest settled F among the pilot records): low = no move (one complete poll);
    mid = 3 moves, each improver in the first block, 1 confirmation each; high = 6 moves, each improver last.
    Exit-then-confirm: poll blocks at the exit projection; EVERY confirmation is one launch at the settled projection
    (confirmations run one per launch, serially): the candidate-confirmation block, the confirmations of each move
    (mid 1; high max(2, the final-poll count below), since an improver ranked last is preceded by every other
    confirm-candidate of its poll), and the confirmations of the FINAL COMPLETE POLL -- every confirm-candidate of the
    poll at the final incumbent is confirmed before the poll completes -- at a confirm-candidate share q of its
    uncached |N| in PROJECTION_FINAL_POLL_CONFIRM_SHARES (0, 1/4, 1/2; the branch is exit-then-confirm only when at most
    1/2 of the screen + ladder rows are not worse, Planner decision P1), ceil(q |N|) confirmations. v1 counted 0 here
    (spec v2 correction, change 9). Settled-direct: poll blocks at the settled projection, no confirmation step.
    -> dict (hours, counts, assumptions; break-even confirmations per complete poll)."""
    import math as _m
    env = LatticeEnv(lattice)
    cached = {tuple(z) for z in screen} | {tuple(z) for z in pool_z}
    unc = sorted(sum(1 for p in env.neighbourhood_points(tuple(z)) if tuple(p['z']) not in cached) for z in screen)
    staged = tuple(pool_z[-1])
    n_st = sum(1 for p in env.neighbourhood_points(staged) if tuple(p['z']) not in cached)
    n_screen = len([z for z in screen if tuple(z) not in {tuple(q) for q in pool_z}])
    pe, ps = projection_h[P.DEPTH_EXIT], projection_h[P.DEPTH_SETTLED]
    common = {'screen_blocks': _m.ceil(n_screen / block), 'ladder_blocks': 1, 'confirmation_blocks': 1}
    b_poll = _m.ceil(n_st / block)
    out = {'assumptions': phase_hours_projection.__doc__, 'uncached_N_screen_plans_sorted': unc,
           'uncached_N_staged': n_st, 'N_staged': len(env.neighbourhood_points(staged)), 'n_screen_uncached': n_screen,
           'per_block_h': dict(projection_h), 'final_poll_confirm_shares': [list(q) for q in
                                                                            PROJECTION_FINAL_POLL_CONFIRM_SHARES],
           'break_even_confirmations_per_complete_poll': {
               'poll_blocks': b_poll, 'value': b_poll * (ps - pe) / ps,
               'rule': 'exit-then-confirm costs b pe + n ps, settled-direct b ps: equal at n = b (ps - pe) / ps '
                       '(one complete poll of the staged plan\'s uncached N)'},
           'cases': {}}
    for cand_name, cn in (('cand_small', unc[:2]), ('cand_large', unc[-2:])):
        poll_blocks = sum(_m.ceil(n / block) for n in cn)
        for sname, n_moves, mv_blocks in (('low', 0, 0), ('mid', 3, 1), ('high', 6, b_poll)):
            srch_blocks = n_moves * mv_blocks + b_poll
            base_e = (common['screen_blocks'] + common['ladder_blocks']) * pe + common['confirmation_blocks'] * ps
            h_sd = base_e + poll_blocks * ps + srch_blocks * ps
            out['cases'][f'{P.BRANCH_SETTLED_DIRECT}|{cand_name}|search_{sname}'] = {
                'hours': h_sd, 'poll_blocks_candidates': poll_blocks, 'search_poll_blocks': srch_blocks,
                'search_confirmation_launches': 0, **common}
            for qn, qd in PROJECTION_FINAL_POLL_CONFIRM_SHARES:
                n_final = _m.ceil(qn * n_st / qd)
                per_move = {'low': 0, 'mid': 1, 'high': max(2, n_final)}[sname]
                n_conf = n_moves * per_move + n_final
                h = base_e + poll_blocks * pe + srch_blocks * pe + n_conf * ps
                out['cases'][f'{P.BRANCH_EXIT_THEN_CONFIRM}|{cand_name}|search_{sname}|q{qn}/{qd}'] = {
                    'hours': h, 'poll_blocks_candidates': poll_blocks, 'search_poll_blocks': srch_blocks,
                    'search_confirmation_launches': n_conf, 'move_confirmations_each': per_move,
                    'final_poll_confirmations': n_final, **common}
    return out


def _split_root_sha(arg):
    """'ROOT@SHA' (CLI form of a declared predecessor) -> (root, sha prefix of >= 8 hex characters)."""
    root, _, sha = str(arg).rpartition('@')
    if not root or len(sha) < 8 or any(c not in '0123456789abcdef' for c in sha):
        raise Refused(f'a declared predecessor is ROOT@SHA (sha256 prefix >= 8 lower-case hex): {arg!r}')
    return root, sha


def verify_unlaunched_predecessor(data_root, root_rel, sha):
    """Planner decision P2: a declared predecessor phase root is UN-LAUNCHED iff it holds exactly one phase spec whose
    name carries its own hash and whose sha256 starts with the declared `sha`, plus at most its freeze logs
    (freeze_stdout.log, freeze_stderr.log), and NOTHING else: no subdirectory at all (no launches/, no eval dirs, no
    interrupted/, stale_locks/ or ceiling/), no state view, no results, no launch log. Files named exactly .DS_Store
    are ignored. -> the record frozen in the spec; raises Refused naming what it found."""
    absr = root_rel if os.path.isabs(root_rel) else os.path.join(data_root, root_rel)
    rel = os.path.relpath(absr, data_root)
    if not os.path.isdir(absr):
        raise Refused(f'declared un-launched predecessor {rel}: no such directory')
    names = sorted(n for n in os.listdir(absr) if n not in IGNORED_FILE_NAMES)
    dirs = [n for n in names if os.path.isdir(os.path.join(absr, n))]
    if dirs:
        raise Refused(f'declared un-launched predecessor {rel} holds directories {dirs} (launches / eval dirs): it is '
                      f'not un-launched; a launched predecessor is declared as a continuation, never excluded')
    specs = [n for n in names if n.startswith('phase_spec_') and n.endswith('.json')]
    if len(specs) != 1:
        raise Refused(f'declared un-launched predecessor {rel}: expected exactly one phase_spec_*.json, found {specs}')
    digest = sha256_file(os.path.join(absr, specs[0]))
    if specs[0] != f'phase_spec_{digest[:8]}.json' or not digest.startswith(str(sha)):
        raise Refused(f'declared un-launched predecessor {rel}: {specs[0]} has sha256 {digest[:16]}..., declared {sha}')
    extra = sorted(set(names) - {specs[0], 'freeze_stdout.log', 'freeze_stderr.log'})
    if extra:
        raise Refused(f'declared un-launched predecessor {rel} holds files other than its spec and freeze logs: '
                      f'{extra}')
    return {'root': rel, 'phase_spec': specs[0], 'spec_sha256': digest, 'files': names,
            'rule': 'un-launched: only the spec and its freeze logs, no directory (.DS_Store ignored); excluded from '
                    'the collision scan, re-verified at every launch'}


def _predecessor_stop(root, n_complete):
    """How a continuation predecessor stopped: exit 5 (results with a review termination) or exit 6 (a ceiling stop
    record after its last complete launch). -> dict; raises Refused otherwise."""
    rp, rm = os.path.join(root, PHASE_RESULTS_FILE), os.path.join(root, PHASE_RESULTS_MANIFEST)
    if os.path.exists(rp):
        if not os.path.exists(rm) or load_json(rm).get(PHASE_RESULTS_FILE) != sha256_file(rp):
            raise Refused(f'continuation predecessor {root}: results without a matching manifest')
        res = load_json(rp)
        reason = ((res.get('report') or {}).get('termination') or {}).get('reason')
        if reason not in P.PHASE_REVIEW_TERMINATIONS or res.get('n_launches') != n_complete:
            raise Refused(f'continuation predecessor {root}: terminated {reason!r} after {res.get("n_launches")} '
                          f'launches -- a continuation resumes a review stop (exit 5) or a ceiling stop (exit 6) only')
        return {'exit': EXIT_REVIEW, 'reason': reason, 'results_sha256': sha256_file(rp)}
    cdir = os.path.join(root, 'ceiling')
    stops = sorted(n for n in os.listdir(cdir) if n.startswith('ceiling_stop_')) if os.path.isdir(cdir) else []
    if stops:
        last = load_json(os.path.join(cdir, stops[-1]))
        if last.get('n_launches_complete') == n_complete:
            return {'exit': EXIT_CEILING, 'reason': 'ceiling', 'ceiling_stop': stops[-1],
                    'ceiling_stop_sha256': sha256_file(os.path.join(cdir, stops[-1])),
                    'ceiling_check': last.get('ceiling_check')}
    raise Refused(f'continuation predecessor {root}: neither stopped for review (exit 5) nor at the ceiling (exit 6) '
                  f'after its last complete launch {n_complete}')


def continuation_source(orc, data_root, root_rel, sha):
    """Planner decision P2: a declared predecessor phase root WITH launches, stopped at exit 5 or 6, as a CACHE ROOT of
    a continuation spec. Its complete launches are verified (contiguous 1..n, none incomplete, each record names the
    predecessor spec, the hash chain links, every manifest hash holds and the manifest is exactly the launch's files);
    its records are indexed by exact eval key under THIS spec's harness configuration (another key: listed as
    ignored); records of interrupted launches (moved aside) are never indexed. No exit view is served from it. Its
    measured hours (`phase_hours`) are carried into the continuation's ceiling. -> the cache source."""
    absr = root_rel if os.path.isabs(root_rel) else os.path.join(data_root, root_rel)
    rel = os.path.relpath(absr, data_root)
    spec_path, digest, pspec = load_phase_spec(absr)
    if not digest.startswith(str(sha)):
        raise Refused(f'continuation predecessor {rel}: spec sha256 {digest[:16]}..., declared {sha}')
    if phase_flex_of(pspec) != phase_flex_of(orc.spec):   # TASKS 5d: a continuation never crosses price levels
        raise Refused(f'continuation predecessor {rel}: price level {phase_flex_of(pspec)} != this spec\'s '
                      f'{phase_flex_of(orc.spec)} (a continuation resumes a phase at its own price level)')
    idx, complete = _complete_indices(absr)
    if not complete:
        raise Refused(f'continuation predecessor {rel} has no complete launch: declare it as an un-launched '
                      f'predecessor instead')
    if set(idx) != set(complete) or complete != list(range(1, len(complete) + 1)):
        raise Refused(f'continuation predecessor {rel}: launches {idx}, complete {complete} (an incomplete launch: '
                      f'resume or move it aside with the predecessor\'s own --phase-run first)')
    entries, ignored, prev = [], [], None
    for n in complete:
        ldir, rec_p, man_p = launch_paths(absr, n)
        lr, man = load_json(rec_p), load_json(man_p)
        rel_rec = os.path.relpath(rec_p, absr)
        bad = []
        if lr.get('phase_spec_sha256') != digest or lr.get('launch_index') != n:
            bad.append('spec sha256 / index')
        if lr.get('previous_launch_manifest_sha256') != prev:
            bad.append('hash chain')
        if man.get(rel_rec) != sha256_file(rec_p):
            bad.append('manifest does not hold the launch record hash')
        if set(man) != {os.path.relpath(f, absr) for f in files_under(ldir)} | {rel_rec}:
            bad.append('manifest file set')
        if any(not os.path.isfile(os.path.join(absr, f)) or sha256_file(os.path.join(absr, f)) != h
               for f, h in man.items()):
            bad.append('a manifest hash does not hold')
        if bad:
            raise Refused([f'continuation predecessor {rel} launch {n}: {b}' for b in bad])
        prev = sha256_file(man_p)
        for r in lr.get('records') or []:
            z, depth = tuple(r['z']), r['depth']
            ed = os.path.join(absr, r['eval_dir'])
            if orc.eval_key(z, depth) != r['eval_key']:
                ignored.append({'eval_dir': os.path.relpath(ed, data_root), 'reason': 'eval key is not this spec\'s '
                                                                                     'key (another configuration)'})
                continue
            entries.append({'z': list(z), 'depth': depth, 'eval_dir': os.path.relpath(ed, data_root),
                            'eval_key': r['eval_key'], 'record_sha256': r['record_sha256'], 'launch': n})
    keys = [(tuple(e['z']), e['depth']) for e in entries]
    if len(set(keys)) != len(keys):
        raise Refused(f'continuation predecessor {rel}: a (z, depth) is indexed twice')
    hours = phase_hours(absr, len(complete))
    return {'root': rel, 'kind': 'continuation_predecessor', 'exit_views': False, 'entries': entries,
            'ignored': ignored,
            'predecessor': {'phase_spec': os.path.basename(spec_path), 'spec_sha256': digest,
                            'phase_id': pspec.get('phase_id'), 'n_launches': len(complete),
                            'last_launch_manifest_sha256': prev, 'stop': _predecessor_stop(absr, len(complete)),
                            'hours': {'cumulative_h': hours['cumulative_h'],
                                      'completed_launches_h': hours['completed_launches_h'],
                                      'interrupted_launches_h': hours['interrupted_launches_h']},
                            'interrupted_records': hours['interrupted']}}


def verify_predecessors_at_launch(spec, data_root):
    """Every --phase-run: the declared un-launched predecessors are still un-launched; each continuation predecessor
    still holds exactly the complete launches indexed at the freeze (none added, none incomplete)."""
    for pr in ((spec.get('predecessors') or {}).get('unlaunched') or []):
        verify_unlaunched_predecessor(data_root, pr['root'], pr['spec_sha256'])
    for src in spec.get('cache_sources') or []:
        if src.get('kind') != 'continuation_predecessor':
            continue
        absr = os.path.join(data_root, src['root'])
        idx, complete = _complete_indices(absr)
        n = src['predecessor']['n_launches']
        if idx != list(range(1, n + 1)) or complete != idx:
            raise Refused(f"continuation predecessor {src['root']}: launches {idx} (complete {complete}) != the {n} "
                          f"indexed at the freeze")


def phase_spec(*, phase_id, oracle, test_only, data_root, predictions=None, test_config=None,
               unlaunched_predecessors=(), continuation_of=(), ceiling_h=None, ceiling_note=None,
               pilot_exit_views=True, flex_price=PHASE_FLEX):
    """Build the phase spec (frozen by `phase_freeze`). Real: oracle 'harness', not test-only, recorded predictions
    Q1-Q7, the pilot root as cache root, the collision scan and the cache-root key audit. Test-only: 'synthetic' or
    'stub_children' oracle with a test config (landscape, knobs, precached pool members, ceiling, simulated walls).
    Spec v2: `unlaunched_predecessors` / `continuation_of` = [(root, sha prefix)] (section comment); a CONTINUATION
    (harness oracles only) may raise the ceiling (`ceiling_h` with `ceiling_note`, the author's decision) and may stop
    serving the pilot exit views (`pilot_exit_views=False`, the R4 fallback: x0, the unit and the staged plan then get
    exit-depth runs, and the exit-view check is recorded, not stopping).
    1x phase (TASKS 5d; section comment at PHASE_FLEX_LEVELS): `flex_price` = 1.0 -- the 1x arm, the 60 h ceiling,
    the predictions Q1-Q9, the harness configuration at 1.0, and NO pilot record (a non-continuation 1x spec MUST pass
    pilot_exit_views=False: x0, the unit and the staged plan are evaluated by the phase; the exit-view check stays
    'stop'); no cache source. `flex_price` = 2.0 (default): exactly the 2x spec as before."""
    import p515_s44_campaign_harness as H
    flex = phase_flex_level(flex_price)
    one_x = flex != PHASE_FLEX
    lattice, inst = build_lattice(data_root)
    tc = copy.deepcopy(test_config or {})
    if oracle not in ORACLES:
        raise Refused(f'oracle {oracle!r}')
    if oracle == 'harness' and (test_only or tc):
        raise Refused('a harness phase is the real phase: not test-only, no test config')
    if oracle != 'harness' and not test_only:
        raise Refused('a synthetic or stub-children phase must be test-only')
    if continuation_of and oracle == 'synthetic':
        raise Refused('a continuation needs a harness oracle (stub_children or harness): its predecessor is a cache root')
    if (ceiling_h is not None or (not pilot_exit_views and not one_x)) and not continuation_of:
        raise Refused('a raised ceiling and --no-pilot-exit-views are for continuation specs only')
    if one_x and pilot_exit_views:
        raise Refused('a 1x phase serves no pilot exit view (no pilot record carries a 1x key; Planner decision D2): '
                      'freeze it with --no-pilot-exit-views')
    if ceiling_h is not None and (not isinstance(ceiling_note, str) or not ceiling_note.strip()):
        raise Refused('a continuation ceiling needs --ceiling-note (the author\'s decision, quoted)')
    if ceiling_h is not None and 'ceiling_h' in tc:
        raise Refused('the ceiling is given twice (test config and --ceiling-h)')
    screen, ladder_all = _screen_and_ladder_tables(lattice)
    x0 = lattice.x0()
    unit = named_start(lattice, 'reference_unit')
    staged = None
    sp, _sha, ps = load_pilot_spec(os.path.join(data_root, PHASE_PILOT_ROOT_REL))
    st = next(e for e in ps['entries'] if e['label'] == 'staged_i1')
    case_rel = H.s56_case_rel(H.validate_derived_instance(ps['configuration'].get('derived_instance')))
    staged = lattice.z_of_canonical(H.s56_staged_canonical(st['options'][H.S56_STAGED_INVESTMENT_KEY], case_rel)[0])
    if staged is None or lattice.label(staged) != PILOT_STAGED_LABEL['srp1grid'] or lattice.reasons(staged):
        raise Refused('the pilot staged plan does not map to its ruled label on the lattice')
    proj = phase_projection_inputs(data_root)
    e_ev = float(tc.get('e_ev', proj['e_EV']))
    if not test_only and round(e_ev) != PHASE_E_EV_ROUNDED:
        raise Refused(f'e_EV {e_ev} != {PHASE_E_EV_ROUNDED} EUR (Planner decision)')
    k_ev = int(tc.get('k_ev', PHASE_K_EV))
    raised = ceiling_h is not None
    level_ceiling = PHASE_CEILING_1X_H if one_x else PHASE_CEILING_H
    ceiling_h = float(ceiling_h if raised else tc.get('ceiling_h', level_ceiling))
    if not test_only and (k_ev != PHASE_K_EV or (ceiling_h != level_ceiling and not raised)):
        raise Refused(f'the real phase uses K_EV 300 and the {level_ceiling:g} h ceiling at price level {flex:g}x (a '
                      f'continuation may raise the ceiling)')
    unl = [verify_unlaunched_predecessor(data_root, *rs) for rs in unlaunched_predecessors]
    spec = {
        'schema': PHASE_SPEC_SCHEMA, 'phase_id': phase_id, 'search_id': phase_id, 'arm': 'ev_1x' if one_x else 'ev_2x',
        'frozen_utc': _utc(), 'authority': PHASE_AUTHORITY_1X if one_x else PHASE_AUTHORITY,
        'test_only': bool(test_only),
        'start': {'name': 'x0', 'z': list(x0), 'label': lattice.label(x0), 'key': lattice.key(x0),
                  'note': 'the start incumbent of the marginal-value table and the efficiency rule'},
        'policy': {'R': P.R_EUR, 'e_ev': e_ev, 'e_floor': e_ev, 'block': P.BLOCK, 'cap': P.CAP,
                   'class_rank': list(P.CLASS_RANK), 'k_ev': k_ev, 'n_max_sub': DOMAIN_SIZE,
                   'n_max_note': f'no evaluation count is declared for the phase: the author\'s {level_ceiling:g} h '
                                 'ceiling is its budget, so each sub-search (candidate polls, the search) has N_max = '
                                 'the domain size (non-binding)',
                   'classification': 'P.classify_exit with e_floor = e_EV (worse dF >= R + e; confirm_candidate '
                                     'dF <= -R + e; else unresolved; e = max(e_EV, bar sum))',
                   'acceptance': 'plan v4.1 section 3 (P.confirm; settled-direct: P.classify_settled)',
                   'not_settled': 'integration ruling 5 (601775c0), per sub-search and phase-wide',
                   'exit_view_check': 'stop' if pilot_exit_views or one_x and not continuation_of else 'record',
                   'exit_view_check_rule': ('Planner decision R4: for each candidate, its exit run\'s (read_cycle, F, '
                                            'bar) == the exit_view of its settled summary, bitwise; mismatch -> exit 5 '
                                            '"exit_view_mismatch" (mode stop) or recorded (mode record: no exit view '
                                            'served)')},
        'phase': {
            'x0': list(x0),
            'screen': screen,
            'ladder': {'zp': list(PHASE_LADDER_ZP), 'n_node_years': PHASE_LADDER_N_NODE_YEARS,
                       'rule': 'P.ladder_plan', 'all_possible_points': ladder_all,
                       'inadmissible_possible_points': [r['label'] for r in ladder_all if r['reasons']]},
            'candidates': {'n': PHASE_N_CANDIDATES, 'rule': 'P.select_candidates (author decision 3)'},
            'efficiency': {'threshold': [1, 2], 'band_eur': e_ev, 'measure': 'not_worse_against_x0',
                           'not_worse_classes': list(P.EFFICIENCY_NOT_WORSE_CLASSES),
                           'rule': 'P.efficiency_rule (author-approved; measure: Planner decision P1, spec v2); the v1 '
                                   '+- e_EV band fraction recorded beside it, information only'},
            'pilot_exit_views': bool(pilot_exit_views),
            'pool': {'fixed': [{'name': 'x0', 'z': list(x0)}, {'name': 'reference_unit', 'z': list(unit)},
                               {'name': 'pilot_staged', 'z': list(staged)}],
                     'cache_members': [], 'rule': 'P.phase_step stage pool'},
            'ceiling': {'ceiling_h': ceiling_h, 'raised_by_continuation': ({'note': ceiling_note} if raised else None),
                        'carried_h': 0.0, 'projection_h': proj['projection_h'], 'projection_inputs': proj,
                        'simulated_walls': bool(test_only and tc.get('simulated_walls', True)),
                        'rule': ('before each block: carried hours (a continuation\'s predecessors\' measured hours, '
                                 'frozen; 0 otherwise) + cumulative measured hours (completed launch walls + '
                                 'interrupted launch walls) + projected wall of the next block (max(frozen projection '
                                 'at its depth, largest measured wall of a completed launch at that depth)) <= '
                                 'ceiling_h; otherwise exit 6 with an hours summary')},
            'exit_codes': {'more_work': EXIT_SUSPENDED, 'complete': EXIT_TERMINATED, 'review': EXIT_REVIEW,
                           'ceiling': EXIT_CEILING, 'refused': EXIT_REFUSED, 'crash': EXIT_FAILED},
            'review_stops': list(P.PHASE_REVIEW_TERMINATIONS)},
        'projected_hours': phase_hours_projection(lattice, [r['z'] for r in screen], [x0, unit, staged],
                                                  proj['projection_h']),
        'instance': inst, 'domain_size': DOMAIN_SIZE, 'oracle': {'kind': oracle}, 'mapping': MAPPING_PARAMETERS,
        'cache_sources': [], 'code_sha256': code_hashes(), 'git_head': _git_head(data_root),
        'interpreter': sys.executable,
        'predecessors': {'unlaunched': unl, 'continuation': [], 'rule': 'Planner decision P2 (spec v2)'}}
    if one_x:   # TASKS 5d: only a 1x spec carries these (a 2x spec keeps its exact format)
        spec['phase']['price_level'] = {
            'flex_price_multiplier': flex, 'levels_allowed': list(PHASE_FLEX_LEVELS),
            'D1': 'e_EV = 17,494 EUR and K_EV = 300 carry over from the 2x pilot; no 1x pilot (the offset review stop '
                  'and the exit-view check Q7 guard the carry-over)',
            'D2': 'no record carries over (the price level is in the eval key): x0 (exit and settled), the reference '
                  'unit and the staged plan (settled) are evaluated by the phase itself on the R4 path '
                  '(pilot_exit_views False); no cache source is declared',
            'exit_view_check': 'stop (Q7 as at 2x; record is for a continuation that serves no exit view)'}
        spec['phase']['ceiling']['projection_restated_1x'] = copy.deepcopy(PHASE_PROJECTION_1X)
    if oracle == 'harness':
        validate_phase_predictions(predictions, PHASE_PREDICTION_IDS_1X if one_x else PHASE_PREDICTION_IDS)
        spec['harness'] = phase_harness_config(data_root, flex)
        spec['capture_path_checklist'] = phase_capture_checklist(H, price_level_1x=one_x)
        spec['predictions_recorded_before_run'] = predictions
        domain = enumerate_domain(lattice)
        orc = HarnessOracle(spec, lattice, lock_path=None)
        if not one_x:
            spec['cache_sources'] = build_cache_index(orc, lattice, data_root, [PHASE_PILOT_ROOT_REL], domain)
            spec['cache_sources'][0].update(kind='pilot', exit_views=bool(pilot_exit_views))
            got = {(tuple(e['z']), e['depth']) for s in spec['cache_sources'] for e in s['entries']}
            want = {(tuple(z), P.DEPTH_SETTLED) for z in (x0, unit, staged)}
            if got != want:
                raise Refused(f'the pilot cache index is not exactly x0 / unit / staged at the settled key: '
                              f'{sorted((lattice.label(z), d) for z, d in got)}')
        spec['phase']['pool']['cache_members'] = []   # no other settled record in the declared cache roots
        spec['cache_exit_views'] = ('the exit summary of each cached settled record is its exit view (map_record at '
                                    'depth exit, read at the would-be certification cycle), served only when no '
                                    'exit-depth record of the plan exists, and only from a cache source whose '
                                    'exit_views is True (the pilot source unless --no-pilot-exit-views)')
        if one_x:
            spec['cache_exit_views'] = ('1x phase: no cache source and no exit view served (D2); every exit value is '
                                        'an exit run, every settled value a settled run of this phase')
        _add_continuations(spec, orc, data_root, continuation_of)
        spec['domain_keys_distinct'] = len({orc.eval_key(z, d) for z in domain for d in P.DEPTHS}) == 2 * DOMAIN_SIZE
        spec['cache_root_key_audit'] = cache_root_key_audit(spec, lattice, data_root, domain, orc)
        if not spec['cache_root_key_audit']['ok']:
            raise Refused(f"cache-root key audit: {spec['cache_root_key_audit']}")
    else:
        land = tc.get('landscape') or {}
        target = [int(v) for v in land.get('target', list(staged))]
        knobs = copy.deepcopy(land.get('knobs') or {})
        pre = [list(x0), list(unit), list(staged)]
        knobs['always_ok'] = [list(z) for z in knobs.get('always_ok', [])] + pre
        knobs['exit_views'] = True   # spec v2: exit summaries carry read_cycle, settled ones their exit view
        spec['oracle'].update({'seed': str(land.get('seed', 'phase')), 'target': target,
                               'alpha': float(land.get('alpha', 30000.0)), 'f0': float(land.get('f0', 810.0e6)),
                               'knobs': knobs, 'precached': [] if one_x else pre,   # 1x: nothing carries over (D2)
                               'note': 'SYNTHETIC VALUES -- not evaluations of the system; x0, the unit and the '
                                       'staged plan precached at both depths (they stand for the pilot records; '
                                       'settled depth only when pilot_exit_views is False)'})
        if oracle == 'stub_children':
            hc = copy.deepcopy(tc['harness'])
            if hc['by_depth']['settled']['cap'] != k_ev:
                raise Refused('stub phase: the settled cap must equal K_EV')
            spec['harness'] = hc
            _add_continuations(spec, HarnessOracle(spec, lattice, lock_path=None), data_root, continuation_of)
    return spec


def _add_continuations(spec, orc, data_root, continuation_of):
    for root_rel, sha in continuation_of:
        src = continuation_source(orc, data_root, root_rel, sha)
        spec['cache_sources'].append(src)
        spec['predecessors']['continuation'].append({'root': src['root'], **src['predecessor']})
    if continuation_of:
        spec['phase']['ceiling']['carried_h'] = sum(c['hours']['cumulative_h']
                                                    for c in spec['predecessors']['continuation'])


def cache_root_key_audit(spec, lattice, data_root, domain, orc):
    """The pilot-report item 4 check on the phase freeze: every eval key the phase can request (domain x 2 depths),
    against the TRACKED artefacts INSIDE the declared cache roots (the freeze scan excludes them). The only keys found
    there must be exactly the indexed cache entries' (x0_i1, unit_i1, staged_i1 settled), and no exit-depth key
    (spec v2: other than an indexed entry of a continuation predecessor, whose screen runs are exit-depth)."""
    keys = {orc.eval_key(z, d) for z in domain for d in P.DEPTHS}
    exit_keys = {orc.eval_key(z, P.DEPTH_EXIT) for z in domain}
    roots = [os.path.realpath(os.path.join(data_root, c['root'])) + os.sep for c in spec['cache_sources']]
    files = [f for f in committed_spec_files(data_root) if any(os.path.realpath(f).startswith(r) for r in roots)]
    found = {}
    for f in files:
        ks = set()
        _collect_eval_keys(load_json(f), ks)
        for k in ks & keys:
            found.setdefault(k, []).append(os.path.relpath(f, data_root))
    expected = {e['eval_key'] for s in spec['cache_sources'] for e in s['entries']}
    cont_exit = {e['eval_key'] for s in spec['cache_sources'] if s.get('kind') == 'continuation_predecessor'
                 for e in s['entries'] if e['depth'] == P.DEPTH_EXIT}
    return {'n_files_in_cache_roots': len(files), 'files': [os.path.relpath(f, data_root) for f in files],
            'keys_found': {k[:16]: v for k, v in sorted(found.items())},
            'expected_cache_keys16': sorted(k[:16] for k in expected),
            'exit_depth_keys_found': sorted(k[:16] for k in set(found) & exit_keys),
            'exit_depth_keys_found_outside_continuation_entries': sorted(k[:16] for k in (set(found) & exit_keys)
                                                                         - cont_exit),
            'ok': set(found) == expected and not ((set(found) & exit_keys) - cont_exit),
            'rule': 'keys of the phase found inside the declared cache roots == exactly the indexed cache entries'}


def phase_freeze(root, spec):
    if os.path.exists(root) and [n for n in os.listdir(root) if not n.endswith('.log')
                                 and n not in IGNORED_FILE_NAMES]:
        raise Refused(f'phase root exists and holds files other than launch logs (write-once): {root}')
    os.makedirs(root, exist_ok=True)
    text = P.canonical_json(spec)
    digest = sha256_text(text)
    path = os.path.join(root, f'phase_spec_{digest[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    if sha256_file(path) != digest:
        raise Refused('frozen phase spec hash mismatch after write')
    return path, digest


def load_phase_spec(root):
    if not os.path.isdir(root):
        raise Refused(f'no phase root {root}')
    names = sorted(n for n in os.listdir(root) if n.startswith('phase_spec_') and n.endswith('.json'))
    if len(names) != 1:
        raise Refused(f'expected exactly one phase_spec_*.json in {root}, found {names}')
    path = os.path.join(root, names[0])
    digest = sha256_file(path)
    if names[0] != f'phase_spec_{digest[:8]}.json':
        raise Refused(f'{names[0]} does not carry its own hash prefix ({digest[:8]})')
    spec = load_json(path)
    if spec.get('schema') != PHASE_SPEC_SCHEMA:
        raise Refused(f'phase spec schema {spec.get("schema")!r}')
    return path, digest, spec


def phase_config(spec):
    pol, ph = spec['policy'], spec['phase']
    return {'phase_id': spec['phase_id'], 'arm': spec['arm'], 'x0': list(ph['x0']),
            'screen': [list(r['z']) for r in ph['screen']], 'ladder_zp': list(ph['ladder']['zp']),
            'ladder_n_node_years': ph['ladder']['n_node_years'], 'n_candidates': ph['candidates']['n'],
            'pool_fixed': copy.deepcopy(ph['pool']['fixed']), 'pool_cache_members': copy.deepcopy(
                ph['pool']['cache_members']), 'R': pol['R'], 'e_ev': pol['e_ev'], 'k_ev': pol['k_ev'],
            'block': pol['block'], 'cap': pol['cap'], 'class_rank': list(pol['class_rank']),
            'n_max_sub': pol['n_max_sub'], 'domain_size': spec['domain_size'],
            'exit_view_check': pol['exit_view_check']}


def phase_cache(spec, oracle, data_root, lattice):
    """The phase's cache records (section comment): the indexed cache entries re-verified and mapped, plus the exit
    views of the cached settled records of the sources whose `exit_views` is True; test-only: the synthetic precached
    pool members (settled depth only when the spec's pilot_exit_views is False)."""
    records = {}
    if spec.get('cache_sources'):
        for k, s in load_cache(spec, oracle, data_root).items():
            records[k] = dict(s, served_as='cache_record')
        for src in spec['cache_sources']:
            if src.get('exit_views') is not True:
                continue
            for e in src['entries']:
                z = tuple(e['z'])
                if e['depth'] != P.DEPTH_SETTLED or (z, P.DEPTH_EXIT) in records:
                    continue
                files = read_eval_files(os.path.join(data_root, e['eval_dir']))
                view = map_record(files, P.DEPTH_EXIT, oracle.mapping)
                records[(z, P.DEPTH_EXIT)] = dict(view, served_as='exit_view_of_settled_cache_record',
                                                  settled_eval_key=e['eval_key'], settled_eval_dir=e['eval_dir'])
    pre = spec['oracle'].get('precached') or []
    if pre:
        if not spec.get('test_only'):
            raise Refused('precached synthetic records on a non-test phase')
        land = synthetic_landscape_of(spec, lattice)
        views = spec['phase'].get('pilot_exit_views', True) is not False
        for z in pre:
            for d in P.DEPTHS:
                if d == P.DEPTH_EXIT and not views:
                    continue
                if (tuple(z), d) in records:
                    raise Refused(f'precached {z} {d} twice')
                records[(tuple(z), d)] = dict(land.summary(z, d), served_as='synthetic_precached')
    return records


def phase_replay(root, spec, spec_sha, lattice, env, oracle, cache):
    """The phase's launch chain re-derived (the search's `replay`, with `P.phase_step`). -> (state, records, chain,
    states). Raises Refused on any disagreement."""
    _idx, complete = _complete_indices(root)
    state = P.phase_initial_state(phase_config(spec))
    records = dict(cache)
    chain, prev_manifest, states = [], None, []
    for n in complete:
        ldir, rec_p, man_p = launch_paths(root, n)
        lr, manifest = load_json(rec_p), load_json(man_p)
        bad = []
        if lr.get('schema') != PHASE_LAUNCH_SCHEMA or lr.get('launch_index') != n:
            bad.append('schema / index')
        if lr.get('phase_spec_sha256') != spec_sha:
            bad.append('phase spec sha256')
        if lr.get('previous_launch_manifest_sha256') != prev_manifest:
            bad.append('hash chain')
        rel_rec = os.path.relpath(rec_p, root)
        if manifest.get(rel_rec) != sha256_file(rec_p):
            bad.append('manifest does not hold the launch record hash')
        expected = {os.path.relpath(f, root) for f in files_under(ldir)} | {rel_rec}
        if set(manifest) != expected:
            bad.append(f'manifest file set differs ({len(set(manifest) ^ expected)} paths)')
        if any(sha256_file(os.path.join(root, f)) != h for f, h in manifest.items()
               if os.path.isfile(os.path.join(root, f))):
            bad.append('a manifest hash does not hold')
        state, action = P.phase_step(state, records, env)
        if P.canonical_json(action) != P.canonical_json(lr.get('action')):
            bad.append(f'replayed action {action} != recorded {lr.get("action")}')
        if P.canonical_json(state) != P.canonical_json(lr.get('state_with_action')):
            bad.append('replayed state != recorded state_with_action')
        states.append(state)
        if action.get('kind') == 'evaluate':
            bad += oracle.verify_launch(ldir, action)
        got = []
        for r in lr.get('records') or []:
            z, depth = tuple(r['z']), r['depth']
            key = oracle.eval_key(z, depth)
            ed = os.path.join(root, r['eval_dir'])
            if key != r['eval_key'] or not os.path.basename(ed).startswith(key[:16]):
                bad.append(f'eval key of {list(z)} {depth} does not recompute')
                continue
            if os.path.dirname(os.path.dirname(ed)) != ldir:
                bad.append(f'eval dir {r["eval_dir"]} is not inside launch {n}')
                continue
            summ, rfile = oracle.summary_from_dir(ed, z, depth)
            if sha256_file(rfile) != r['record_sha256'] or P.canonical_json(summ) != P.canonical_json(r['summary']):
                bad.append(f'record of {list(z)} {depth} does not re-derive from its eval dir')
            got.append(((z, depth), summ))
        if action.get('kind') != 'evaluate' or sorted(P.canonical_json([list(k[0]), k[1]]) for k, _ in got) != \
                sorted(P.canonical_json([z, action['depth']]) for z in action['zs']):
            bad.append('the launch records are not exactly its action')
        if bad:
            raise Refused([f'launch {n}: {b}' for b in bad])
        for k, s in got:
            if k in records:
                raise Refused(f'launch {n}: record {k} already present (write-once)')
            records[k] = s
        chain.append({'n': n, 'manifest_sha256': sha256_file(man_p), 'record_sha256': manifest[rel_rec]})
        prev_manifest = chain[-1]['manifest_sha256']
    allowed_top = {os.path.basename(load_phase_spec(root)[0]), 'launches', 'interrupted', 'stale_locks', 'ceiling',
                   PHASE_STATE_VIEW, PHASE_RESULTS_FILE, PHASE_RESULTS_MANIFEST}
    foreign = [n for n in os.listdir(root) if n not in allowed_top and not n.endswith('.log')
               and n not in IGNORED_FILE_NAMES]
    if foreign:
        raise Refused(f'foreign files in the phase root: {foreign}')
    return state, records, chain, states


def phase_hours(root, n_complete):
    """Measured machine hours of the phase so far (section comment)."""
    per = []
    for n in range(1, n_complete + 1):
        lr = load_json(launch_paths(root, n)[1])
        per.append({'launch': n, 'depth': lr['action']['depth'], 'purpose': lr['action']['purpose'],
                    'n_evaluations': len(lr['action']['zs']), 'wall_s': lr['wall_s'],
                    'accounted_wall_s': lr['accounted_wall_s']})
    inter = []
    idir = os.path.join(root, 'interrupted')
    for name in sorted(os.listdir(idir)) if os.path.isdir(idir) else []:
        if not (name.startswith('interrupted_') and name.endswith('.json')):
            continue
        rec = load_json(os.path.join(idir, name))
        st = [k['started_t'] for k in rec['killed_evaluations'] if isinstance(k.get('started_t'), (int, float))]
        lw = [k['last_write_t'] for k in rec['killed_evaluations'] if isinstance(k.get('last_write_t'), (int, float))]
        inter.append({'record': name, 'wall_h': (max(lw) - min(st)) / 3600.0 if st and lw else 0.0})
    done_h = sum(p['accounted_wall_s'] for p in per) / 3600.0
    int_h = sum(i['wall_h'] for i in inter)
    return {'completed_launches_h': done_h, 'interrupted_launches_h': int_h, 'cumulative_h': done_h + int_h,
            'per_launch': per, 'interrupted': inter}


def ceiling_check(spec, hours, depth):
    ce = spec['phase']['ceiling']
    measured = [p['accounted_wall_s'] / 3600.0 for p in hours['per_launch'] if p['depth'] == depth]
    proj = max([ce['projection_h'][depth]] + measured)
    carried = float(ce.get('carried_h', 0.0))
    total = carried + hours['cumulative_h'] + proj
    return {'cumulative_h': hours['cumulative_h'], 'carried_h': carried, 'projected_next_block_h': proj,
            'next_block_depth': depth,
            'frozen_projection_h': ce['projection_h'][depth], 'max_measured_same_depth_h': max(measured, default=None),
            'cumulative_plus_projected_h': total, 'ceiling_h': ce['ceiling_h'], 'stop': total > ce['ceiling_h'],
            'rule': ce['rule']}


def _phase_code(state):
    return EXIT_REVIEW if state['termination']['reason'] in P.PHASE_REVIEW_TERMINATIONS else EXIT_TERMINATED


def _phase_view(root, chain, state):
    atomic_json(os.path.join(root, PHASE_STATE_VIEW), {'n_launches': len(chain), 'state': state,
                                                       'state_sha256': sha256_text(P.canonical_json(state)),
                                                       'derived': 'replayed from the launch chain'})


def phase_prediction_inputs(spec, report, root, data_root, chain, records):
    """The measured quantities the recorded predictions Q1-Q6 read (no verdict here: the evaluation is the
    Planner's / a Worker's, against the frozen statements)."""
    table = (report.get('marginal_value_table') or {})
    rows = table.get('rows') or []
    scr = [r for r in rows if r['source'] == 'screen']
    he = sum(1 for s in records.values() if s.get('status') == 'harness_error')
    lsf, rss, n_rec = 0, [], 0
    for n in range(1, len(chain) + 1):
        lr = load_json(launch_paths(root, n)[1])
        for r in lr['records']:
            n_rec += 1
            rp = os.path.join(root, r['eval_dir'], 'evaluation_record.json')
            if spec['oracle']['kind'] == 'harness' and os.path.isfile(rp):
                rec = load_json(rp)
                lsf += int(rec.get('local_solve_failures') or 0)
                b = ((rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss'))
                if isinstance(b, (int, float)):
                    rss.append(b / GIB)
    pool = report.get('pool') or {}
    out = {
        'Q1': {'screen_rows': len(scr), 'screen_certified': sum(1 for r in scr if r['status'] == 'certified'),
               'screen_dF_exit_below_0': sum(1 for r in scr if isinstance(r['dF_exit'], (int, float))
                                             and r['dF_exit'] < 0),
               'screen_dF_exit_below_minus_R': sum(1 for r in scr if isinstance(r['dF_exit'], (int, float))
                                                   and r['dF_exit'] < -spec['policy']['R']),
               'max_screen_dF_exit': max((r['dF_exit'] for r in scr if isinstance(r['dF_exit'], (int, float))),
                                         default=None)},
        'Q2': {'harness_error_summaries': he, 'local_solve_failures_total': lsf if spec['oracle']['kind'] == 'harness'
               else None, 'records_read': n_rec},
        'Q3': {'best_start': pool.get('chosen'), 'best_start_label': pool.get('chosen_label'),
               'best_start_is_x0': pool.get('chosen_label') == 'x0'},
        'Q4': {'termination': report.get('termination'), 'cumulative_h': phase_hours(root, len(chain))['cumulative_h'],
               'ceiling_h': spec['phase']['ceiling']['ceiling_h']},
        'Q5': {'peak_child_rss_gib': max(rss) if rss else None, 'n_records_with_rss': len(rss),
               'ceiling_gib': PHASE_MEMORY_PER_SLOT_GIB},
        'Q6': report.get('efficiency'),
        'Q7': {'mode': spec['policy'].get('exit_view_check'),
               'candidates': [{'label': c['label'], 'exit_view_check': c.get('exit_view_check')}
                              for c in (report.get('candidates') or [])],
               'n_checked': sum(1 for c in (report.get('candidates') or []) if c.get('exit_view_check')),
               'n_equal': sum(1 for c in (report.get('candidates') or [])
                              if (c.get('exit_view_check') or {}).get('equal') is True),
               'termination_reason': (report.get('termination') or {}).get('reason')},
        'carried_h': spec['phase']['ceiling'].get('carried_h', 0.0),
    }
    if spec['phase'].get('price_level'):   # the 1x phase (TASKS 5d): Q8 (computed after the phase), Q9 (the leak)
        st = sorted((('|'.join(map(str, z)), s) for (z, d), s in records.items() if d == P.DEPTH_SETTLED),
                    key=lambda t: t[0])
        lk = [((s.get('leak') or {}).get('priced_loss_eur'), zk) for zk, s in st]
        vals = [v for v, _zk in lk if _fin(v)]
        ceil = (spec.get('mapping') or MAPPING_PARAMETERS).get('leak_ceiling_eur')
        out['Q8'] = {'price_level': spec['phase']['price_level'].get('flex_price_multiplier'),
                     'screen_dF_exit': {r['label']: r['dF_exit'] for r in scr},
                     'note': 'the 2x values and the mean difference are computed after the phase at zero solves '
                             '(the predictions file names the 2x results file and its sha256)'}
        out['Q9'] = {'n_settled_summaries': len(st), 'n_with_leak': len(vals),
                     'max_priced_loss_eur': max(vals) if vals else None, 'ceiling_eur': ceil,
                     'n_above_ceiling': sum(1 for v in vals if _fin(ceil) and v > ceil),
                     'above_ceiling_z': [zk for v, zk in lk if _fin(v) and _fin(ceil) and v > ceil],
                     'without_leak_z': [zk for v, zk in lk if not _fin(v)]}
    return out


def _write_phase_results(root, spec, spec_sha, state, records, chain, hooks, data_root):
    rep = P.phase_terminal_report(state)
    payload = {'schema': PHASE_RESULTS_SCHEMA, 'phase_id': spec['phase_id'], 'phase_spec_sha256': spec_sha,
               'oracle': spec['oracle'], 'n_launches': len(chain),
               'last_launch_manifest_sha256': chain[-1]['manifest_sha256'] if chain else None,
               'report': rep, 'hours': phase_hours(root, len(chain)),
               'prediction_inputs': phase_prediction_inputs(spec, rep, root, data_root, chain, records),
               'state': state,
               'objective_convention': ('F = I(x) + gross_operational_cost - TV credit (oracle mapping '
                                        f'{MAPPING_SCHEMA}), EUR of total discounted planning cost'
                                        if spec['oracle']['kind'] == 'harness' else
                                        'SYNTHETIC values (synthetic / stub-children oracle)')}
    res = os.path.join(root, PHASE_RESULTS_FILE)
    sha = write_once_json(res, payload)
    hooks.at('after_results')
    write_once_json(os.path.join(root, PHASE_RESULTS_MANIFEST), {PHASE_RESULTS_FILE: sha})
    _log(f'phase terminated: {state["termination"]["reason"]}; results written')


def phase_run(root, *, data_root, lock_path, take_over_stale, preflight_only_out=None):
    """ONE phase launch = one block (section comment). -> exit code. `preflight_only_out` (a NEW directory): zero
    solves -- the lock, the replay, the next action, the ceiling check and `real_launch_preflight`, recorded there;
    nothing launched, nothing written under the phase root. Spec v2: first the code-drift refusal and the predecessor
    re-verification (exit 2, nothing written)."""
    spec_path, spec_sha, spec = load_phase_spec(root)
    drift = code_drift(spec)
    if drift:
        raise Refused([f"code drift: {d['file']} sha256 now {str(d['now'])[:16]}..., frozen in the spec "
                       f"{str(d['frozen'])[:16]}... -- the phase runs only the code its spec froze" for d in drift])
    verify_predecessors_at_launch(spec, data_root)
    lattice, _inst = build_lattice(data_root)
    env = LatticeEnv(lattice)
    oracle = make_oracle(spec, lattice, lock_path)
    idx, complete = _complete_indices(root)
    n_next = len(complete) + 1
    pf = preflight_only_out is not None
    if pf:
        if spec['oracle']['kind'] != 'harness':
            raise Refused('--preflight-only is for the real (harness) phase')
        if os.path.exists(preflight_only_out):
            raise Refused(f'--out exists (write-once): {preflight_only_out}')
        if set(idx) != set(complete):
            raise Refused('an incomplete launch exists: a real launch would move it aside first; the preflight-only '
                          'mode moves nothing')
    hooks = _NoHooks() if pf else KillHooks(spec, n_next)
    lock, _moved = acquire_lock(lock_path, spec['phase_id'], n_next, root, take_over_stale)
    if isinstance(oracle, HarnessOracle):
        oracle.lock_content = lock
    try:
        hooks.at('after_lock')
        cache = phase_cache(spec, oracle, data_root, lattice)
        if not pf:
            handle_interruptions(root, spec, oracle.kind, (PHASE_RESULTS_FILE, PHASE_RESULTS_MANIFEST))
        state, records, chain, states = phase_replay(root, spec, spec_sha, lattice, env, oracle, cache)
        state, action = P.phase_step(state, records, env)
        view_p = os.path.join(root, PHASE_STATE_VIEW)
        if os.path.exists(view_p):
            view = load_json(view_p)
            v = view.get('n_launches')
            if not isinstance(v, int) or not 0 <= v <= len(chain):
                raise Refused(f'the state view names launch {v}; the chain holds {len(chain)}')
            replayed = state if v == len(chain) else states[v]
            if P.canonical_json(view.get('state')) != P.canonical_json(replayed):
                raise Refused(f'the state view (launch {v}) does not equal the replayed state')
        if action['kind'] == 'terminated':
            if pf:
                raise Refused(f"the phase has terminated ({state['termination']['reason']}): nothing to preflight")
            _phase_view(root, chain, state)
            if not os.path.exists(os.path.join(root, PHASE_RESULTS_MANIFEST)):
                _write_phase_results(root, spec, spec_sha, state, records, chain, hooks, data_root)
            _log(f"phase already terminated: {state['termination']['reason']}")
            return _phase_code(state)
        hours = phase_hours(root, len(chain))
        cc = ceiling_check(spec, hours, action['depth'])
        ldir, rec_p, man_p = launch_paths(root, n_next)
        if pf:
            rec = {'schema': 'p515_s56_phase_preflight_only_v1', 'utc': _utc(),
                   'phase_root': os.path.relpath(root, data_root), 'phase_spec_sha256': spec_sha,
                   'next_launch_index': n_next, 'next_action': action, 'stage': state['stage'],
                   'cache_records': sorted([lattice.label(z), d, s.get('served_as'), s.get('status')]
                                           for (z, d), s in cache.items()),
                   'previous_launch_manifest_sha256': chain[-1]['manifest_sha256'] if chain else None,
                   'launch_dir_would_be': os.path.relpath(ldir, data_root), 'lock_acquired': lock,
                   'ceiling_check': cc, 'hours': hours, 'git_head': _git_head(data_root), 'code_sha256': code_hashes(),
                   'solves': 0, 'launched': False}
            try:
                rec['real_launch_preflight'] = real_launch_preflight(
                    oracle.H, ldir, min(int(spec['harness']['concurrency']), len(action['zs'])),
                    spec['harness']['memory_per_slot_gib'], data_root, mapping=spec.get('mapping'))
                rec['pass'] = not cc['stop']
            except Refused as error:
                rec['real_launch_preflight'] = {'refusals': error.reasons}
                rec['pass'] = False
            rec['launch_dir_created'] = os.path.exists(ldir)
            os.makedirs(preflight_only_out)
            rsha = write_once_json(os.path.join(preflight_only_out, 'phase_preflight.json'), rec)
            _log(f"phase preflight-only (launch {n_next}: {action['purpose']} {action['depth']} x{len(action['zs'])}):"
                 f" pass {rec['pass']}; ceiling {cc['cumulative_plus_projected_h']:.2f} / {cc['ceiling_h']} h; "
                 f"record sha256 {rsha}")
            if not rec['pass']:
                raise Refused((rec['real_launch_preflight'] or {}).get('refusals') or ['ceiling'])
            return EXIT_TERMINATED
        if cc['stop']:
            cdir = os.path.join(root, 'ceiling')
            os.makedirs(cdir, exist_ok=True)
            m = 1 + len([n for n in os.listdir(cdir) if n.startswith('ceiling_stop_')])
            summary = {'schema': 'p515_s56_phase_ceiling_stop_v1', 'utc': _utc(), 'index': m,
                       'n_launches_complete': len(chain), 'next_launch_would_be': n_next, 'next_action': action,
                       'stage': state['stage'], 'ceiling_check': cc, 'hours': hours,
                       'note': f"the author decided the {spec['phase']['ceiling']['ceiling_h']:g} h ceiling; the "
                               f"phase is suspended, not terminated"}
            write_once_json(os.path.join(cdir, f'ceiling_stop_{m:03d}.json'), summary)
            _phase_view(root, chain, state)
            _log(f"CEILING: cumulative {cc['cumulative_h']:.2f} h + projected next block {cc['projected_next_block_h']:.2f}"
                 f" h ({action['depth']}) = {cc['cumulative_plus_projected_h']:.2f} h > {cc['ceiling_h']} h; "
                 f"{len(chain)} launches complete; stage {state['stage']}; nothing launched")
            return EXIT_CEILING
        preflight = None
        if isinstance(oracle, HarnessOracle) and not oracle.stub:
            preflight = real_launch_preflight(oracle.H, ldir, min(int(spec['harness']['concurrency']),
                                                                  len(action['zs'])),
                                              spec['harness']['memory_per_slot_gib'], data_root,
                                              mapping=spec.get('mapping'))
        started, started_utc = time.time(), _utc()
        os.makedirs(ldir)
        hooks.at('after_launch_dir')
        produced = oracle.evaluate(action, ldir, hooks, launch_index=n_next)
        hooks.at('after_evaluate')
        recs = []
        for r in produced:
            summ, rfile = oracle.summary_from_dir(r['eval_dir'], r['z'], r['depth'])
            recs.append({'z': r['z'], 'depth': r['depth'], 'eval_key': r['eval_key'],
                         'eval_dir': os.path.relpath(r['eval_dir'], root), 'record_file': os.path.relpath(rfile, root),
                         'record_sha256': sha256_file(rfile), 'summary': summ})
        wall = time.time() - started
        sim = spec['phase']['ceiling'].get('simulated_walls') is True
        sim_walls = [r['summary'].get('wall_s') for r in recs if _fin(r['summary'].get('wall_s'))]
        accounted = max(sim_walls) if (sim and sim_walls) else wall
        lr = {'schema': PHASE_LAUNCH_SCHEMA, 'launch_index': n_next, 'phase_id': spec['phase_id'],
              'phase_spec_sha256': spec_sha, 'previous_launch_manifest_sha256': chain[-1]['manifest_sha256']
              if chain else None, 'action': action, 'state_with_action': state, 'records': recs,
              'started_utc': started_utc, 'finished_utc': _utc(), 'wall_s': wall,
              'accounted_wall_s': accounted, 'accounted_wall_rule': ('simulated: max record wall_s (test-only)'
                                                                     if sim and sim_walls else 'measured launch wall'),
              'ceiling_check_before_launch': cc, 'pid': os.getpid(),
              'driver_sha256': sha256_file(os.path.abspath(__file__)), 'policy_sha256': sha256_file(P.__file__),
              'oracle': oracle.kind, **({'real_launch_preflight': preflight} if preflight is not None else {})}
        rec_sha = write_once_json(rec_p, lr)
        hooks.at('after_launch_record')
        manifest = {os.path.relpath(f, root): sha256_file(f) for f in files_under(ldir)}
        manifest[os.path.relpath(rec_p, root)] = rec_sha
        man_sha = write_once_json(man_p, manifest)
        hooks.at('after_manifest')
        chain.append({'n': n_next, 'manifest_sha256': man_sha, 'record_sha256': rec_sha})
        for r in recs:
            records[(tuple(r['z']), r['depth'])] = r['summary']
        state2, action2 = P.phase_step(state, records, env)
        _phase_view(root, chain, state2)
        hooks.at('after_state_view')
        _log(f"launch {n_next}: stage {action['stage']} {action['purpose']} {action['depth']} x{len(action['zs'])}; "
             f"wall {wall / 3600.0:.3f} h; cumulative {hours['cumulative_h'] + accounted / 3600.0:.2f} h; next: "
             f"{action2['kind']}" + (f" ({action2['termination']['reason']})" if action2['kind'] == 'terminated'
                                     else f" {action2['purpose']} {action2['depth']}"))
        if action2['kind'] == 'terminated':
            _write_phase_results(root, spec, spec_sha, state2, records, chain, hooks, data_root)
            return _phase_code(state2)
        return EXIT_SUSPENDED
    finally:
        release_lock(lock_path)


CANONICAL_PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'   # CLAUDE.local.md


def phase_author_command(repo, root_rel, python=CANONICAL_PYTHON, extra_args=''):
    """The author's single Terminal command: a foreground loop (never detached) under `caffeinate -i` that re-issues
    `--phase-run` while the exit code is 4. Each invocation logs to <root>/launch_NNN_stdout.log / _stderr.log (NNN =
    the invocation counter, continued from the logs already present) and prints its exit code; any code other than 4
    ends the loop (0 complete, 5 review, 6 ceiling, other = fault). `--take-over-stale-lock` resumes after a reboot or
    a kill: the driver takes a lock over only when its pid is dead and it names this phase. `extra_args`: tests only
    (a scratch --lock-path)."""
    loop = (f'R={root_rel}; i=1; while [ -e "$R/launch_$(printf %03d $i)_stdout.log" ]; do i=$((i+1)); done; '
            f'while true; do n=$(printf %03d $i); NLP_SOLVER_PATH=/usr/local/bin/ipopt {python} -u '
            f'p515_s56_search_driver.py --phase-run --root "$R" --take-over-stale-lock{extra_args} '
            f'> "$R/launch_${{n}}_stdout.log" 2> "$R/launch_${{n}}_stderr.log"; rc=$?; '
            f'echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; '
            f'if [ $rc -ne 4 ]; then echo "loop stopped: exit $rc (0 complete, 5 review, 6 ceiling, other fault)"; '
            f'break; fi; i=$((i+1)); done')
    return f"cd {repo} && caffeinate -i zsh -c '{loop}'"


def phase_single_launch_command(repo, root_rel, n=1, python=CANONICAL_PYTHON, extra_args=''):
    """Spec v2, the first-block hold (Planner decision R2/P2): ONE invocation of --phase-run, logged exactly as the
    loop logs invocation `n` (launch_NNN_stdout.log / _stderr.log), so that `phase_author_command` afterwards continues
    the numbering at n + 1. Refuses to run (prints, runs nothing) if that invocation's stdout log already exists."""
    one = (f'R={root_rel}; n={n:03d}; if [ -e "$R/launch_${{n}}_stdout.log" ]; then echo "launch_${{n}} logs exist: '
           f'not re-run"; else NLP_SOLVER_PATH=/usr/local/bin/ipopt {python} -u p515_s56_search_driver.py --phase-run '
           f'--root "$R" --take-over-stale-lock{extra_args} > "$R/launch_${{n}}_stdout.log" 2> '
           f'"$R/launch_${{n}}_stderr.log"; rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $n exit $rc"; fi')
    return f"cd {repo} && caffeinate -i zsh -c '{one}'"


PHASE_CHECK_SCRIPT = 's56_phase_check.py'   # its name and its command line must not contain the harness hazard string


def phase_check_command(repo, root_rel, launch=1, python=CANONICAL_PYTHON):
    """The author's check of one launch record (`s56_phase_check.py`, stdlib only, read-only)."""
    return f'cd {repo} && {python} {PHASE_CHECK_SCRIPT} {root_rel} --launch {launch}'


def phase_in_process(spec, lattice, max_actions=100000):
    """Zero solves, synthetic specs only: the whole phase in process (no files) -> (state, records). Used by the tests
    as the reference the launch machinery must reproduce byte for byte."""
    if spec['oracle']['kind'] not in ('synthetic', 'stub_children'):
        raise Refused('in-process phase: synthetic landscapes only')
    env = LatticeEnv(lattice)
    land = synthetic_landscape_of(spec, lattice)
    records = phase_cache(dict(spec, cache_sources=[]), None, None, lattice)
    state = P.phase_initial_state(phase_config(spec))
    for _ in range(max_actions):
        state, action = P.phase_step(state, records, env)
        if action['kind'] == 'terminated':
            return state, records
        for z in action['zs']:
            k = (tuple(z), action['depth'])
            if k in records:
                raise Refused(f'in-process phase: {k} delivered twice')
            records[k] = land.summary(z, action['depth'])
    raise Refused('in-process phase: max_actions reached')


# ======================================================================================================================
#  CLI
# ======================================================================================================================
def _freeze_search(args, data_root, root):
    lattice, inst = build_lattice(data_root)
    class_rank = P.CLASS_RANK_REVERSED if args.reversed_class_rank else P.CLASS_RANK
    hc = load_json(args.harness_config) if args.harness_config else None
    common = dict(search_id=args.search_id, arm=args.arm, lattice=lattice, instance_rec=inst, n_max=args.n_max,
                  k_ev=args.k_ev, oracle=args.oracle, class_rank=class_rank, test_only=args.test_only,
                  data_root=data_root, synthetic_seed=args.synthetic_seed, harness_config=hc)
    start = 'x0' if args.start == 'best_of_pool' else args.start
    spec = build_spec(start_name=start, **common)
    if (args.cache_root or args.start == 'best_of_pool') and args.oracle == 'synthetic':
        raise Refused('cache roots and the start pool are for harness oracles')
    if args.cache_root:
        oracle = HarnessOracle(spec, lattice, lock_path=None)
        spec['cache_sources'] = build_cache_index(oracle, lattice, data_root, args.cache_root)
    if args.start == 'best_of_pool':
        if not args.cache_root:
            raise Refused('--start best_of_pool needs --cache-root (the screen / start-pool records)')
        pool = [{'name': n, 'z': list(named_start(lattice, n))} for n in ('x0', 'reference_unit')]
        for m in (load_json(args.start_pool) if args.start_pool else []):
            pool.append({'name': m['name'], 'z': [int(v) for v in m['z']]})
        records = load_cache(spec, HarnessOracle(spec, lattice, lock_path=None), data_root)
        z, rep = best_start(lattice, pool, records)
        cache = spec['cache_sources']
        spec = build_spec(start_name='best_of_pool', start_z=z, start_report=rep, **common)
        spec['cache_sources'] = cache
    if args.oracle in ('stub_children', 'harness'):
        spec['collision_scan'] = freeze_collision_scan(spec, lattice, root, data_root)
    return freeze(root, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--pilot-freeze', action='store_true', help='freeze a pilot root (no evaluation)')
    mode.add_argument('--pilot-run', action='store_true', help='one pilot launch = the next block of <= 5')
    mode.add_argument('--pilot-evaluate', action='store_true', help='zero solves: the pilot outputs')
    mode.add_argument('--project', action='store_true', help='zero solves: the wall-time projection')
    mode.add_argument('--phase-freeze', action='store_true', help='freeze a 2x phase root (no evaluation)')
    mode.add_argument('--phase-run', action='store_true', help='one phase launch = the next block of <= 5')
    ap.add_argument('--root', help='search / pilot root (absolute or relative to --data-root)')
    ap.add_argument('--out', help='--dry-run / --pilot-evaluate output directory (new)')
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--search-id')
    ap.add_argument('--arm', default='ev_2x')
    ap.add_argument('--start', default='x0', choices=sorted(NAMED_STARTS) + ['best_of_pool'])
    ap.add_argument('--start-pool', help='JSON list of {name, z}: the declared pool members beyond x0 and the unit')
    ap.add_argument('--cache-root', action='append', default=[], help='a declared cache root (repeatable)')
    ap.add_argument('--n-max', type=int)
    ap.add_argument('--k-ev', type=int)
    ap.add_argument('--oracle', default='synthetic', choices=ORACLES)
    ap.add_argument('--synthetic-seed', default='s1')
    ap.add_argument('--reversed-class-rank', action='store_true')
    ap.add_argument('--test-only', action='store_true')
    ap.add_argument('--lock-path', default=None, help='test-only specs: a scratch lock path')
    ap.add_argument('--take-over-stale-lock', action='store_true')
    ap.add_argument('--harness-config', help='--freeze, harness oracles: JSON file of the per-depth harness '
                                             'configuration (embedded verbatim in the search spec)')
    ap.add_argument('--pilot-id')
    ap.add_argument('--grid', choices=PILOT_GRIDS)
    ap.add_argument('--memory-per-slot-gib', type=float)
    ap.add_argument('--s-share-eur', type=float, help='--pilot-evaluate: S\'s share of R/4; the frozen spec\'s value '
                                                       '(ruling 2: 500) is used when absent, a different value refused')
    ap.add_argument('--predictions', help='--pilot-freeze (real): JSON file {"predictions": {PP1..PP8}} embedded '
                                          'verbatim in the pilot spec')
    ap.add_argument('--grid-note', help='--pilot-freeze: free text recorded with the grid (e.g. its authority)')
    ap.add_argument('--preflight-only', action='store_true',
                    help='--pilot-run: zero solves; the next block\'s real-launch preflight only, record to --out')
    ap.add_argument('--phase-id')
    ap.add_argument('--phase-test-config', help='--phase-freeze, test-only phases: JSON file (landscape, knobs, '
                                                'ceiling_h, k_ev, harness for stub children)')
    ap.add_argument('--unlaunched-predecessor', action='append', default=[],
                    help='--phase-freeze (spec v2): ROOT@SHA of a declared predecessor phase root verified un-launched '
                         '(excluded from the collision scan); repeatable')
    ap.add_argument('--continuation-of', action='append', default=[],
                    help='--phase-freeze (spec v2): ROOT@SHA of a predecessor phase root with launches, stopped at exit '
                         '5 or 6, declared as a cache root of this continuation spec; repeatable')
    ap.add_argument('--ceiling-h', type=float, help='--phase-freeze, continuation only: the raised ceiling (author)')
    ap.add_argument('--ceiling-note', help='--phase-freeze with --ceiling-h: the author\'s decision, quoted')
    ap.add_argument('--no-pilot-exit-views', action='store_true',
                    help='--phase-freeze, continuation (or a 1x phase, where it is required): serve no pilot exit view '
                         '(the R4 fallback)')
    ap.add_argument('--phase-flex', type=float, default=PHASE_FLEX,
                    help=f'--phase-freeze: the phase\'s price level, one of {PHASE_FLEX_LEVELS} (default {PHASE_FLEX}; '
                         f'1.0 = the 1x phase, TASKS 5d)')
    ap.add_argument('--t-exit-h', type=float)
    ap.add_argument('--t-set-h', type=float)
    ap.add_argument('--psi', type=float)
    ap.add_argument('--blocks', type=int)
    ap.add_argument('--confirmation-batches', type=int)
    args = ap.parse_args(argv)
    data_root = os.path.abspath(args.data_root)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.15 S56 search driver parent (never solves)').install()
    try:
        if args.dry_run:
            out = args.out if os.path.isabs(args.out or '') else os.path.join(data_root, args.out)
            dry_run(out, data_root)
            code = EXIT_TERMINATED
        elif args.project:
            print(json.dumps(project_wall_time(args.t_exit_h, args.t_set_h, args.psi, args.blocks,
                                               args.confirmation_batches), indent=1))
            code = EXIT_TERMINATED
        else:
            root = args.root if os.path.isabs(args.root) else os.path.join(data_root, args.root)
            if args.freeze:
                path, digest = _freeze_search(args, data_root, root)
                _log(f'frozen {path} ({digest})')
                code = EXIT_TERMINATED
            elif args.pilot_freeze:
                pred = None
                if args.predictions:
                    pp = args.predictions if os.path.isabs(args.predictions) else os.path.join(data_root,
                                                                                                args.predictions)
                    pred = dict(load_json(pp), source_file_sha256=sha256_file(pp))
                spec = pilot_spec(pilot_id=args.pilot_id, grid=args.grid, oracle=args.oracle,
                                  test_only=args.test_only, data_root=data_root,
                                  memory_per_slot_gib=args.memory_per_slot_gib, predictions=pred,
                                  grid_note=args.grid_note)
                path, digest = pilot_freeze(root, spec)
                _log(f'pilot frozen {path} ({digest}): blocks {spec["blocks"]}')
                code = EXIT_TERMINATED
            elif args.phase_freeze:
                pred, tc = None, None
                if args.predictions:
                    pp = args.predictions if os.path.isabs(args.predictions) else os.path.join(data_root,
                                                                                                args.predictions)
                    pred = dict(load_json(pp), source_file_sha256=sha256_file(pp))
                if args.phase_test_config:
                    tc = load_json(args.phase_test_config)
                spec = phase_spec(phase_id=args.phase_id, oracle=args.oracle, test_only=args.test_only,
                                  data_root=data_root, predictions=pred, test_config=tc,
                                  unlaunched_predecessors=[_split_root_sha(a) for a in args.unlaunched_predecessor],
                                  continuation_of=[_split_root_sha(a) for a in args.continuation_of],
                                  ceiling_h=args.ceiling_h, ceiling_note=args.ceiling_note,
                                  pilot_exit_views=not args.no_pilot_exit_views, flex_price=args.phase_flex)
                if args.oracle in ('stub_children', 'harness'):
                    lat_f = build_lattice(data_root)[0]
                    spec['collision_scan'] = freeze_collision_scan(spec, lat_f, root, data_root)
                path, digest = phase_freeze(root, spec)
                _log(f'phase frozen {path} ({digest})')
                code = EXIT_TERMINATED
            elif args.pilot_evaluate:
                out = args.out if os.path.isabs(args.out or '') else os.path.join(data_root, args.out)
                res = pilot_evaluate(root, out, args.s_share_eur, data_root)
                _log(f"pilot evaluated: K_EV {res.get('K_EV')} e_EV {res.get('e_EV')} failures {res['failures']}")
                code = EXIT_TERMINATED if not res['failures'] else EXIT_REVIEW
            else:
                spec = (load_pilot_spec(root)[2] if args.pilot_run else load_phase_spec(root)[2] if args.phase_run
                        else load_spec(root)[2])
                if args.lock_path and not spec.get('test_only'):
                    raise Refused('--lock-path is for test-only specs')
                lock_path = args.lock_path or os.path.join(data_root, '.p515_s44_campaign.lock')
                if not args.lock_path and os.path.exists(os.path.join(data_root, '.p515_g_gate.lock')):
                    raise Refused('the legacy one-run lock .p515_g_gate.lock exists')
                if args.phase_run:
                    pf_out = None
                    if args.preflight_only:
                        if not args.out:
                            raise Refused('--preflight-only needs --out (a new directory)')
                        pf_out = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
                    elif args.out:
                        raise Refused('--out is for --preflight-only')
                    code = phase_run(root, data_root=data_root, lock_path=lock_path,
                                     take_over_stale=args.take_over_stale_lock, preflight_only_out=pf_out)
                elif args.pilot_run:
                    pf_out = None
                    if args.preflight_only:
                        if not args.out:
                            raise Refused('--preflight-only needs --out (a new directory)')
                        pf_out = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
                    elif args.out:
                        raise Refused('--out is for --preflight-only on --pilot-run')
                    code = pilot_run(root, data_root=data_root, lock_path=lock_path,
                                     take_over_stale=args.take_over_stale_lock, preflight_only_out=pf_out)
                else:
                    code = run_launch(root, data_root=data_root, lock_path=lock_path,
                                      take_over_stale=args.take_over_stale_lock)
    except Refused as error:
        for r in error.reasons:
            _log(f'REFUSED: {r}')
        code = EXIT_REFUSED
    fails = guard.verify(0)
    if fails:
        _log(f'SOLVE GUARD FAILED: {fails}')
        code = EXIT_FAILED
    return code


if __name__ == '__main__':
    sys.exit(main())
