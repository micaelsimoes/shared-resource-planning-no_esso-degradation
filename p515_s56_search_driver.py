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
stop_for_review_offset, start not certified); 4 launch done, more work remains (re-issue the same command); 2 refused
(nothing written); 1 crash. ONE launch = one action, attached, both streams captured, never detached; for unattended
progress the author re-issues in a foreground loop under `caffeinate -i`. One search at a time per checkout (the
harness campaign lock).

ORACLES (--oracle at --freeze; recorded in the spec):
  synthetic      in-process synthetic landscape (dry runs and the gate); writes one record file per evaluation
  stub_children  the harness `evaluate` with stub children (gate only; values from the synthetic landscape)
  harness        real evaluations -- NOT PERMITTED in this task; refused unless the spec is frozen for it (later)

--dry-run (zero solves, guard armed): the full search against the synthetic oracle in process, for a named start;
writes nothing but its own output directory (a NEW directory).
"""

import argparse
import copy
import hashlib
import json
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
REVIEW_TERMINATIONS = ('harness_error', 'stop_for_review_offset', 'start_incumbent_not_certified')
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


def files_under(path):
    out = []
    for directory, _dirs, files in os.walk(path):
        out += [os.path.join(directory, f) for f in files]
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

    def __init__(self, seed, target, start, alpha=30000.0, f0=810.0e6):
        self.seed, self.target, self.start = str(seed), tuple(target), tuple(start)
        self.alpha, self.f0 = float(alpha), float(f0)

    def describe(self):
        return {'seed': self.seed, 'target': list(self.target), 'start': list(self.start), 'alpha': self.alpha,
                'f0': self.f0, 'text': SyntheticLandscape.__doc__}

    def _u(self, z, tag):
        h = sha256_text(f'{self.seed}|{tag}|{" ".join(str(v) for v in z)}')
        return int(h[:13], 16) / float(1 << 52)

    def summary(self, z, depth):
        z = tuple(z)
        d2 = sum((a - b) ** 2 for a, b in zip(z, self.target))
        f_exit = self.f0 + self.alpha * d2 + (self._u(z, 'noise') - 0.5) * 6000.0
        if depth == P.DEPTH_EXIT:
            status = 'certified' if (z == self.start or self._u(z, 'cert') >= 0.05) else 'not_certified'
            return {'depth': P.DEPTH_EXIT, 'status': status, 'F': f_exit,
                    'bar': 3600.0 + self._u(z, 'bar') * 26000.0, 'wall_s': 3600.0 * (1.0 + self._u(z, 'wall'))}
        s = 2400.0 if self._u(z, 'vfail') < 0.10 else 200.0 + self._u(z, 'S') * 400.0
        return {'depth': P.DEPTH_SETTLED, 'status': 'settled', 'F': f_exit + (self._u(z, 'off') - 0.5) * 2000.0,
                'V': {'S': s, 'PR': self._u(z, 'PR') * 500.0, 'G': self._u(z, 'G') * 300.0,
                      'A': self._u(z, 'A') * 1000.0},
                'gates_pass': self._u(z, 'gate') >= 0.03, 'wall_s': 7200.0 * (1.0 + self._u(z, 'wall'))}


def synthetic_landscape_of(spec, lattice):
    o = spec['oracle']
    return SyntheticLandscape(o['seed'], o['target'], spec['start']['z'], o.get('alpha', 30000.0),
                              o.get('f0', 810.0e6))


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


class HarnessOracle:
    """The campaign harness: one frozen harness spec per launch (its own campaign root = the launch dir), its entries
    the action's candidates at the action's depth, evaluated by ONE `H.evaluate` call (concurrency from the search
    spec). Every eval key the harness freezes is recomputed here (`eval_key`) and must agree. Stub children
    ('stub_children'): the harness stub mode (no model import), values from the synthetic landscape, attached only to
    a stub record of the same candidate. Real records ('harness'): refused in this build -- the objective field, the
    settled-run V components and the gates are not yet named (report: Blocked on Planner)."""

    def __init__(self, spec, lattice, lock_path):
        import p515_s44_campaign_harness as H
        self.H, self.spec, self.lattice, self.lock_path = H, spec, lattice, lock_path
        self.kind = spec['oracle']['kind']
        self.stub = self.kind == 'stub_children'
        if not self.stub:
            raise Refused('the harness oracle (real evaluations) is not available in this build')
        self.land = synthetic_landscape_of(spec, lattice)
        self.hz = spec['harness']
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
        H = self.H
        cfg = self.hz['by_depth'][depth]['configuration']
        opts = self.entry_options(z, depth)
        ck = H.candidate_key(self.lattice.canonical_of_z(tuple(z)))
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
        return H.s56_settled_stop_evaluation_key(ekey, opts.get(H.S56_SETTLED_STOP_DISABLED_KEY))

    def _ctx(self, launch_dir, spec_path, sha, hspec, launch_index):
        H = self.H
        lock = self.lock_path
        raw = os.environ.get(STUB_DELAY_ENV)
        if raw and not self.spec.get('test_only'):
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
        raise Refused('real-record summaries are not built')


def make_oracle(spec, lattice, lock_path=None):
    kind = spec['oracle']['kind']
    if kind == 'synthetic':
        return SyntheticOracle(spec, lattice)
    if kind == 'stub_children':
        return HarnessOracle(spec, lattice, lock_path)
    if kind == 'harness':
        raise Refused('oracle harness: real evaluations are not permitted in this build (no objective mapping)')
    raise Refused(f'unknown oracle {kind!r}')


# ======================================================================================================================
#  freeze
# ======================================================================================================================
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
               data_root, synthetic_seed=None, synthetic_target=None, cache_sources=(), harness_config=None):
    if oracle not in ORACLES:
        raise Refused(f'oracle {oracle!r} not in {ORACLES}')
    if oracle == 'harness':
        raise Refused('the harness oracle is not permitted in this build (no real evaluation, no search launch)')
    if not isinstance(n_max, int) or n_max < 0 or not isinstance(k_ev, int) or k_ev < 1:
        raise Refused('N_max (int >= 0) and K_EV (int >= 1) must be declared at freeze (set after the pilot)')
    z0 = named_start(lattice, start_name)
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
        'start': {'name': start_name, 'z': list(z0), 'label': lattice.label(z0), 'key': lattice.key(z0)},
        'policy': {'R': P.R_EUR, 'e_floor': P.E_FLOOR_EUR, 'block': P.BLOCK, 'cap': P.CAP,
                   'class_rank': list(class_rank), 'n_max': n_max, 'k_ev': k_ev,
                   'acceptance': 'plan v4.1 section 3 (P.confirm)', 'classification': 'P.classify_exit'},
        'instance': instance_rec, 'domain_size': DOMAIN_SIZE,
        'oracle': {'kind': oracle},
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


def collision_scan(files, keys, own_root, allowed_roots=()):
    """Pre-run key-collision check over artefacts (campaign / search specs, launch records): every `eval_key` they
    carry is compared with `keys`. Files under `own_root` (the search's own root) and under the declared cache roots
    are EXCLUDED -- a scan that included them would match the artefacts this search itself wrote (CLAUDE.md, the
    W105 rule). -> {'n_files_scanned', 'excluded_own_root', 'excluded_cache_roots', 'collisions', 'refusals'}."""
    own = os.path.realpath(own_root) + os.sep
    allowed = [os.path.realpath(r) + os.sep for r in allowed_roots]
    keys = set(keys)
    out = {'n_files_scanned': 0, 'excluded_own_root': [], 'excluded_cache_roots': [], 'collisions': [],
           'refusals': []}
    for f in files:
        real = os.path.realpath(f)
        if real.startswith(own):
            out['excluded_own_root'].append(f)
            continue
        if any(real.startswith(a) for a in allowed):
            out['excluded_cache_roots'].append(f)
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


def freeze_collision_scan(spec, lattice, root, data_root):
    """At --freeze (harness oracles): every eval key the search can request (the domain x {exit, settled} under the
    spec's harness configuration) against the tracked campaign / search specs and launch records, the search's own
    root and its declared cache roots excluded. A collision refuses the freeze."""
    oracle = HarnessOracle(spec, lattice, lock_path=None)
    keys = {oracle.eval_key(z, d) for z in enumerate_domain(lattice) for d in P.DEPTHS}
    if len(keys) != 2 * DOMAIN_SIZE:
        raise Refused(f'{len(keys)} distinct eval keys over the domain x 2 depths != {2 * DOMAIN_SIZE}')
    res = collision_scan(committed_spec_files(data_root), keys, root,
                         allowed_roots=[os.path.join(data_root, c['root']) for c in spec.get('cache_sources') or []])
    if res['refusals']:
        raise Refused(res['refusals'])
    return {'n_keys': len(keys), 'n_files_scanned': res['n_files_scanned'],
            'n_excluded_own_root': len(res['excluded_own_root']),
            'n_excluded_cache_roots': len(res['excluded_cache_roots']), 'collisions': res['collisions'],
            'scope': 'git ls-files data/SRP1/Results: campaign_spec_* / search_spec_* / launches/launch_*.json'}


def committed_spec_files(data_root):
    """Tracked campaign / search specs and launch records under data/SRP1/Results (the collision scan's scope)."""
    r = subprocess.run(['git', 'ls-files', '--', os.path.join('data', 'SRP1', 'Results')], cwd=data_root,
                       capture_output=True, text=True, check=True)
    names = [n for n in r.stdout.splitlines()
             if os.path.basename(n).startswith(('campaign_spec_', 'search_spec_'))
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


def handle_interruptions(root, spec, oracle_kind):
    """Move every incomplete launch (and results without their manifest) aside, recording the killed evaluations
    with their hours. -> the interruption record (or None)."""
    idx, complete = _complete_indices(root)
    if complete != list(range(1, len(complete) + 1)):
        raise Refused(f'complete launches {complete} are not contiguous 1..n')
    incomplete = [n for n in idx if n not in complete]
    if incomplete and incomplete != [len(complete) + 1]:
        raise Refused(f'incomplete launches {incomplete}: only launch {len(complete) + 1} may be incomplete')
    paths = []
    for n in incomplete:
        paths += [p for p in launch_paths(root, n) if os.path.exists(p)]
    res, man = os.path.join(root, RESULTS_FILE), os.path.join(root, RESULTS_MANIFEST)
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
                                     else 'set at the harness integration')}


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
              'policy_sha256': sha256_file(P.__file__), 'oracle': oracle.kind}
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
#  CLI
# ======================================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    ap.add_argument('--root', help='search root (absolute or relative to --data-root)')
    ap.add_argument('--out', help='--dry-run output directory (new)')
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--search-id')
    ap.add_argument('--arm', default='ev_2x')
    ap.add_argument('--start', default='x0', choices=sorted(NAMED_STARTS))
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
    args = ap.parse_args(argv)
    data_root = os.path.abspath(args.data_root)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.15 S56 search driver parent (never solves)').install()
    try:
        if args.dry_run:
            out = args.out if os.path.isabs(args.out or '') else os.path.join(data_root, args.out)
            dry_run(out, data_root)
            code = EXIT_TERMINATED
        else:
            root = args.root if os.path.isabs(args.root) else os.path.join(data_root, args.root)
            if args.freeze:
                lattice, inst = build_lattice(data_root)
                spec = build_spec(search_id=args.search_id, arm=args.arm, lattice=lattice, instance_rec=inst,
                                  start_name=args.start, n_max=args.n_max, k_ev=args.k_ev, oracle=args.oracle,
                                  class_rank=P.CLASS_RANK_REVERSED if args.reversed_class_rank else P.CLASS_RANK,
                                  test_only=args.test_only, data_root=data_root, synthetic_seed=args.synthetic_seed,
                                  harness_config=load_json(args.harness_config) if args.harness_config else None)
                if args.oracle in ('stub_children', 'harness'):
                    spec['collision_scan'] = freeze_collision_scan(spec, lattice, root, data_root)
                path, digest = freeze(root, spec)
                _log(f'frozen {path} ({digest})')
                code = EXIT_TERMINATED
            else:
                _p, _s, spec = load_spec(root)
                if args.lock_path and not spec.get('test_only'):
                    raise Refused('--lock-path is for test-only specs')
                lock_path = args.lock_path or os.path.join(data_root, '.p515_s44_campaign.lock')
                if not args.lock_path and os.path.exists(os.path.join(data_root, '.p515_g_gate.lock')):
                    raise Refused('the legacy one-run lock .p515_g_gate.lock exists')
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
