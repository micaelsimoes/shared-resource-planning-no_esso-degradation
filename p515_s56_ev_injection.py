"""P5.15 S56 -- expected-profile (EV) instance: the injection hook and the in-child assertion (importable; no guard,
no I/O and no production import at module import time).

WHY THIS EXISTS (S56_DESIGN_NOTE_v1.md, "Additional requirements found", item 1)
  Production has no reader for explicit scenario profiles: it synthesizes every scenario array at read time. The EV
  case file is the SRP1-3x3 case with every scenario count set to 1 -- which is SRP1.json's own content -- so a
  plain production read of it yields the PUBLISHED DRAW 0, not the means. The means reach the run only through the
  injection hook below. `assert_ev_arrays` is the armed guard against the EV case silently running as draw 0: a
  harness child calls it after its read, and it RAISES unless every injected array equals the stored mean bit for
  bit.

THE ARRAYS (the definition of audit A, `p515_s56_expected_profiles._array_refs`, reproduced here so that a harness
child can import it without importing that script, which installs a blocking solve guard at import; the S56 build
script checks the two enumerations are identical, key for key and in order):
  per (year, day) the market arrays planning.cost_energy_p[year][day], planning.cost_flex[year][day]
  (n_market x 24); per network block of the TSO and of every DSO, per load (index in network.loads) pd, qd,
  flexibility.active_power.upward / .downward, flexibility.reactive_power.upward / .downward; per generator
  (index in network.generators) pg, qg (n_operation x 24).
  Key string: market|<energy|flex>|<year>|<day>;  <TSO|DSO<node>>|<year>|<day>|load|<i>|<field>;
              <TSO|DSO<node>>|<year>|<day>|gen|<i>|<gen_type>|<pg|qg>.

A5 (design note v1, D2: "A5 keeps production's constant"): the reactive-flexibility arrays (flex_q_up, flex_q_down)
  are NOT injected. Production sets them to np.ones(shape) * EQUALITY_TOLERANCE (network.py, the load loop of
  `_update_network_with_operational_data`), a tolerance, not data; their probability-weighted mean differs from that
  constant in the last bit. The hook leaves them as production built them and checks they equal the constant; the
  assertion checks the same.

POINTS
  'checksum'  : at the entry of shared_resources_planning._compute_scenario_metadata (the injection point; the
                scenario checksum is computed over the injected arrays).
  'post_read' : after SharedResourcesPlanning.read_planning_problem returns. Production then has replaced the TSO
                ADN loads' pd / qd by np.zeros (shared_resources_planning._add_adn_node_to_transmission_network); the
                profiles file stores the expected post-read value of those keys ('post_read_expected'), computed as
                the mean of the stochastic read's post-read arrays.
"""
import hashlib
import json

import numpy as np

SCHEMA = 'p515_s56_ev_profiles_v1'
A5_FIELDS = ('flex_q_up', 'flex_q_down')
POINTS = ('checksum', 'post_read')


class EvInjectionMismatch(RuntimeError):
    """Raised when the arrays in a planning object are not the stored EV means (or the A5 constant)."""

    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


# ======================================================================================================================
#  enumeration (audit A's definition)
# ======================================================================================================================
def holders(planning):
    out = [('TSO', planning.transmission_network)]
    out += [(f'DSO{node}', planning.distribution_networks[node]) for node in sorted(planning.distribution_networks)]
    return out


def array_refs(planning):
    """Ordered list of (key, kind, live ndarray reference, probability vector, baseMVA); identical to
    p515_s56_expected_profiles._array_refs."""
    refs = []
    for year in sorted(planning.years):
        w = np.asarray(planning.prob_market_scenarios[year], dtype=float)
        for day in planning.days:
            refs.append((('market', 'energy', year, day), 'market', planning.cost_energy_p[year][day], w, None))
            refs.append((('market', 'flex', year, day), 'market', planning.cost_flex[year][day], w, None))
    for hname, holder in holders(planning):
        for year in sorted(holder.years):
            for day in holder.days:
                net = holder.network[year][day]
                w = np.asarray(net.prob_operation_scenarios, dtype=float)
                for i, load in enumerate(net.loads):
                    base = (hname, year, day, 'load', i)
                    refs.append((base + ('pd',), 'operation', load.pd, w, net.baseMVA))
                    refs.append((base + ('qd',), 'operation', load.qd, w, net.baseMVA))
                    refs.append((base + ('flex_p_up',), 'operation', load.flexibility.active_power.upward, w, net.baseMVA))
                    refs.append((base + ('flex_p_down',), 'operation', load.flexibility.active_power.downward, w, net.baseMVA))
                    refs.append((base + ('flex_q_up',), 'operation', load.flexibility.reactive_power.upward, w, net.baseMVA))
                    refs.append((base + ('flex_q_down',), 'operation', load.flexibility.reactive_power.downward, w, net.baseMVA))
                for i, gen in enumerate(net.generators):
                    base = (hname, year, day, 'gen', i, int(gen.gen_type))
                    refs.append((base + ('pg',), 'operation', gen.pg, w, net.baseMVA))
                    refs.append((base + ('qg',), 'operation', gen.qg, w, net.baseMVA))
    return refs


def key_str(key):
    return '|'.join(str(k) for k in key)


def is_a5(key):
    """key: tuple or key string."""
    last = key.rsplit('|', 1)[-1] if isinstance(key, str) else key[-1]
    return last in A5_FIELDS


def production_a5_constant(shape):
    """What production builds for a reactive-flexibility array: np.ones(shape) * EQUALITY_TOLERANCE (network.py),
    with the EQUALITY_TOLERANCE network.py itself resolves."""
    import network as NET   # production module; imported lazily (the child has it loaded already)
    return np.ones(shape) * NET.EQUALITY_TOLERANCE


def _bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.dtype == np.float64 and b.dtype == np.float64 and a.shape == b.shape
            and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


# ======================================================================================================================
#  the profiles file
# ======================================================================================================================
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_profiles(path, expected_sha256=None):
    """Read an EV profiles file (schema SCHEMA). Raises on a schema or sha256 mismatch, or on an A5 key among the
    injected arrays. Returns a dict with ndarrays (1 x T, float64)."""
    sha = _sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise EvInjectionMismatch(f'profiles file {path}: sha256 {sha} != expected {expected_sha256}',
                                  {'profiles_sha256': sha, 'expected_sha256': expected_sha256})
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get('schema') != SCHEMA:
        raise EvInjectionMismatch(f'profiles file {path}: schema {payload.get("schema")!r} != {SCHEMA!r}',
                                  {'profiles_sha256': sha})
    arrays = {k: np.asarray(v, dtype=np.float64).reshape(1, -1) for k, v in payload['arrays'].items()}
    post = {k: np.asarray(v, dtype=np.float64).reshape(1, -1) for k, v in payload['post_read_expected'].items()}
    a5 = list(payload['a5_keys'])
    bad = [k for k in arrays if is_a5(k)] + [k for k in a5 if not is_a5(k)]
    if bad or set(a5) & set(arrays) or not set(post) <= set(arrays):
        raise EvInjectionMismatch(f'profiles file {path}: inconsistent key sets (first {bad[:5]})',
                                  {'profiles_sha256': sha})
    return {'path': path, 'sha256': sha, 'arrays': arrays, 'post_read_expected': post, 'a5_keys': a5,
            'ev_scenario_checksum': payload.get('ev_scenario_checksum'), 'instance_label': payload.get('instance_label')}


# ======================================================================================================================
#  the injection hook (pass-through wrapper on the production checksum point)
# ======================================================================================================================
class ScenarioPointHook:
    """Pass-through wrapper on shared_resources_planning._compute_scenario_metadata.

    profiles=None: capture only (calls `on_point(planning)` if given, then the original).
    profiles=<load_profiles(...)>: inject -- writes every stored mean IN PLACE into the live 1 x T arrays production
      has just generated (aliases such as network.cost_energy_p -> planning.cost_energy_p[y][d] therefore see the
      means), leaves the A5 arrays as production built them (checked equal to the constant), checks the read's key
      set is exactly the file's (injected + A5), then `on_point`, then the original (the checksum is computed over the
      injected arrays)."""

    def __init__(self, profiles=None, on_point=None):
        import shared_resources_planning as SRP
        self._srp = SRP
        self.original = SRP._compute_scenario_metadata
        self.profiles = profiles
        self.on_point = on_point
        self.calls = 0
        self.inject_counts = []
        self.installed = False

    def install(self):
        hook = self

        def wrapped(planning_problem):
            hook.calls += 1
            if hook.profiles is not None:
                hook.inject_counts.append(inject(planning_problem, hook.profiles))
            if hook.on_point is not None:
                hook.on_point(planning_problem)
            return hook.original(planning_problem)
        self._srp._compute_scenario_metadata = wrapped
        self.installed = True
        return self

    def uninstall(self):
        self._srp._compute_scenario_metadata = self.original
        self.installed = False


def inject(planning, profiles):
    """Write the stored means in place (A5 arrays untouched, checked). Returns {'n_injected', 'n_a5_kept'}."""
    arrays, a5 = profiles['arrays'], set(profiles['a5_keys'])
    refs = array_refs(planning)
    keys = [key_str(k) for k, *_ in refs]
    if len(set(keys)) != len(keys):
        raise EvInjectionMismatch('inject: duplicate keys in the read', {})
    if set(keys) != set(arrays) | a5:
        raise EvInjectionMismatch(
            f'inject: the read has {len(keys)} arrays, the file {len(arrays)} + {len(a5)} A5; only in the read '
            f'{sorted(set(keys) - set(arrays) - a5)[:5]}, only in the file {sorted((set(arrays) | a5) - set(keys))[:5]}',
            {})
    n_inj, n_a5 = 0, 0
    for (key, _kind, arr, _w, _b), ks in zip(refs, keys):
        if ks in a5:
            if not _bitwise_equal(arr, production_a5_constant(np.shape(arr))) or np.shape(arr)[0] != 1:
                raise EvInjectionMismatch(f'inject: A5 array {ks} is not the production constant (1 x T)', {'key': ks})
            n_a5 += 1
            continue
        m = arrays[ks]
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float64 or arr.shape != m.shape:
            raise EvInjectionMismatch(f'inject {ks}: live array {type(arr).__name__} {np.shape(arr)} cannot take mean '
                                      f'{m.shape}', {'key': ks})
        arr[...] = m
        n_inj += 1
    return {'n_injected': n_inj, 'n_a5_kept': n_a5}


# ======================================================================================================================
#  the in-child assertion
# ======================================================================================================================
def assert_ev_arrays(planning, profiles, point='post_read', expected_scenario_checksum=None, max_report=20):
    """RAISES EvInjectionMismatch unless, in `planning` at `point` ('checksum' or 'post_read'):
      - the scenario dimensions are 1 (planning.num_market_scenarios, every holder's num_oper_scenarios) and every
        probability vector is [1.0] (planning.prob_market_scenarios[year], every block's prob_market_scenarios and
        prob_operation_scenarios);
      - the key set of the scenario arrays is exactly the file's (injected + A5);
      - every injected array equals the stored mean bit for bit (float64, shape 1 x T) -- at 'post_read' the TSO ADN
        load pd / qd are compared with 'post_read_expected' instead;
      - every A5 array equals production's constant np.ones((1, T)) * EQUALITY_TOLERANCE bit for bit;
      - every network block's own cost_energy_p / cost_flex (what the block model reads) equals the stored market
        mean bit for bit;
      - if `expected_scenario_checksum` is given: planning.scenario_metadata['combined_scenario_checksum'] equals it.
    Returns the summary dict when everything holds."""
    if point not in POINTS:
        raise ValueError(f'point must be one of {POINTS}, got {point!r}')
    arrays, post, a5 = profiles['arrays'], profiles['post_read_expected'], set(profiles['a5_keys'])
    failures = []
    # dimensions and probabilities
    dims = {'num_market_scenarios': getattr(planning, 'num_market_scenarios', None),
            'num_oper_scenarios': {h: holder.num_oper_scenarios for h, holder in holders(planning)}}
    if dims['num_market_scenarios'] != 1 or any(v != 1 for v in dims['num_oper_scenarios'].values()):
        failures.append(f'scenario dimensions are not 1: {dims}')
    if any(list(planning.prob_market_scenarios[y]) != [1.0] for y in planning.years):
        failures.append('planning.prob_market_scenarios is not [1.0] in every year')
    bad_prob = [f'{h}|{y}|{d}' for h, holder in holders(planning) for y in holder.years for d in holder.days
                if list(holder.network[y][d].prob_market_scenarios) != [1.0]
                or list(holder.network[y][d].prob_operation_scenarios) != [1.0]]
    if bad_prob:
        failures.append(f'{len(bad_prob)} network blocks without [1.0] probabilities (first {bad_prob[:3]})')
    # arrays
    refs = array_refs(planning)
    keys = [key_str(k) for k, *_ in refs]
    if len(set(keys)) != len(keys) or set(keys) != set(arrays) | a5:
        failures.append(f'key set differs: read {len(keys)} (distinct {len(set(keys))}), file {len(arrays)} + {len(a5)} '
                        f'A5; only in the read {sorted(set(keys) - set(arrays) - a5)[:3]}, only in the file '
                        f'{sorted((set(arrays) | a5) - set(keys))[:3]}')
    mism, n_inj, n_a5 = [], 0, 0
    for (key, _kind, arr, _w, _b), ks in zip(refs, keys):
        if ks in a5:
            n_a5 += 1
            if not _bitwise_equal(arr, production_a5_constant((1, np.shape(arr)[-1]))):
                mism.append({'key': ks, 'what': 'A5 array is not the production constant (1 x T)'})
            continue
        if ks not in arrays:
            continue
        n_inj += 1
        exp = post[ks] if (point == 'post_read' and ks in post) else arrays[ks]
        if not _bitwise_equal(arr, exp):
            a = np.asarray(arr, dtype=float)
            d = (float(np.max(np.abs(a - exp))) if a.shape == exp.shape else None)
            mism.append({'key': ks, 'what': 'array != stored mean', 'shape': list(np.shape(arr)), 'max_abs_diff': d})
    if mism:
        failures.append(f'{len(mism)} of {n_inj + n_a5} arrays differ from the stored EV profiles (first: {mism[0]})')
    # what each block model reads
    alias_bad = []
    for h, holder in holders(planning):
        for y in holder.years:
            for d in holder.days:
                net = holder.network[y][d]
                for fld, tag in (('cost_energy_p', 'energy'), ('cost_flex', 'flex')):
                    ks = key_str(('market', tag, y, d))
                    if ks not in arrays or not _bitwise_equal(getattr(net, fld), arrays[ks]):
                        alias_bad.append(f'{h}|{y}|{d}|{fld}')
    if alias_bad:
        failures.append(f'{len(alias_bad)} network-block market arrays differ from the stored means (first '
                        f'{alias_bad[:3]})')
    got = (getattr(planning, 'scenario_metadata', None) or {}).get('combined_scenario_checksum')
    if expected_scenario_checksum is not None and got != expected_scenario_checksum:
        failures.append(f'combined scenario checksum {got} != expected {expected_scenario_checksum}')
    summary = {'point': point, 'profiles_sha256': profiles.get('sha256'), 'n_arrays_read': len(keys),
               'n_injected_checked': n_inj, 'n_a5_checked': n_a5, 'n_mismatches': len(mism),
               'mismatches_first': mism[:max_report], 'n_network_market_arrays_bad': len(alias_bad),
               'scenario_dimensions': dims, 'combined_scenario_checksum': got,
               'expected_scenario_checksum': expected_scenario_checksum, 'failures': failures, 'ok': not failures}
    if failures:
        raise EvInjectionMismatch('EV injection assertion FAILED: ' + ' | '.join(failures), summary)
    return summary
