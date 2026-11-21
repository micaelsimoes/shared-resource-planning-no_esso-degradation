"""P5.15 S56 -- dependence stress-test instance (author decision 3, TASKS.md S56): comonotone re-pairing of the
operation scenarios of each network; the derivation, the injection hook and the in-child assertion (importable; no
guard, no I/O and no production import at module import time).

WHY THIS EXISTS
  In the production 3 x 3 instances every network's operation scenarios are drawn independently per (year, day)
  (network_data._read_network_data: block seed derive_random_seed(<network seed>, 'realization', year, day); then, in
  network._update_network_with_operational_data, one `DataFrame.sample` per device AND per quantity, each with its own
  derived seed: load pc / qc / pc_flex / pc_flex_up / pc_flex_down, generator pg / qg), and paired ACROSS networks by
  index: the joint scenario s_o takes scenario s_o of the TSO and of every DSO. Market scenarios form a full product
  with operation scenarios (production _scenario_probability = pi_m * pi_o), so only the cross-network pairing of the
  operation scenarios carries dependence. This module re-labels, within each network and each (year, day) block, that
  network's operation scenarios by rank of a declared daily statistic, so that scenario index 0 is the lowest-statistic
  day in every network and the last index the highest (comonotone pairing). It is a DEPENDENCE STRESS TEST, not a
  calibrated or worst-case joint model.

WITHIN-NETWORK COHERENCE (what "a scenario" is, read from the code)
  Because every quantity has its own seed, row s of one array and row s of another array of the same network are
  independent draws from the synthetic pools (even pc and qc of the same load, which the copula generates jointly, are
  sampled with different seeds). A "scenario" of a network is therefore the index s itself: the tuple of row s of all of
  that network's operation arrays. The re-labelling permutes the scenario index of ALL of that network block's
  operation arrays together (every load's pd, qd, flexibility P up / down, flexibility Q up / down; every generator's
  pg, qg), so each scenario's tuple moves as a unit and the within-network structure, whatever it is, is preserved.
  Market arrays are never permuted.

STATISTIC (declared)
  Daily net demand of network block (h, year, day) in scenario s, in MWh:
      ND[s] = sum_{loads l not ADN} sum_t pd_l[s, t] * baseMVA * dt  -  sum_{generators g with gen_type in
              definitions.GEN_CURTAILLABLE_TYPES} sum_t pg_g[s, t] * baseMVA * dt,      dt = 24 h / num_instants (= 1 h)
  pd is the stored active demand (production: (pc + pc_flex) / baseMVA, i.e. including the flexible-load baseline);
  pg of a curtailable (RES) generator is its available power. "ADN loads" are the TSO loads production replaces by
  the DSO interface right after the checksum point (shared_resources_planning._add_adn_node_to_transmission_network:
  for each DSO node n with network.adn_load_exists(n), the load network.get_adn_load_idx(n) gets pd = qd = 0); they are
  excluded because the model never sees their sampled values. In SRP1 every TSO load is an ADN load, so the TSO
  statistic is minus the TSO's daily RES availability. Evaluation: per array float(np.sum(A[s])) * baseMVA * dt,
  accumulated in Python floats in the declared order (loads by index, then generators by index); deterministic.
  The alternative (TSO ADN loads included at their pre-zeroing sampled values) is computed by `daily_net_demand(...,
  include_adn_loads=True)` for reporting only.

TIE RULE (declared): new index k takes original index perm[k], perm = sorted(range(n), key=lambda s: (ND[s], s)),
  i.e. ascending ND, exact ties broken by the lower original index first. new_A = old_A[perm, :] for every operation
  array of the block.

THE PROFILES FILE (schema SCHEMA) stores, per (network, year, day) block, the permutation and the statistic in the
  original order; per operation-array key the sha256 (ARRAY_SHA_DEFINITION) of the base array and of the permuted
  array at the checksum point; per market key the sha256 of the (unchanged) market array; the sha256 the TSO ADN pd /
  qd take after the read (zeros); the probability vectors. The arrays themselves are not stored: the hook permutes the
  arrays production has just generated, after verifying that they are the base instance's bit for bit.

POINTS
  'checksum'  : entry of shared_resources_planning._compute_scenario_metadata (the injection point, as for the EV twin
                in p515_s56_ev_injection; the production scenario checksum is computed over the permuted arrays).
  'post_read' : after SharedResourcesPlanning.read_planning_problem returns (TSO ADN pd / qd zeroed by production).

KEYS: those of p515_s56_ev_injection.array_refs (audit A's enumeration): market|<energy|flex>|<year>|<day>;
  <TSO|DSO<node>>|<year>|<day>|load|<i>|<pd|qd|flex_p_up|flex_p_down|flex_q_up|flex_q_down>;
  <TSO|DSO<node>>|<year>|<day>|gen|<i>|<gen_type>|<pg|qg>.  Block key: <TSO|DSO<node>>|<year>|<day>.
"""
import hashlib
import json

import numpy as np

import p515_s56_ev_injection as EVI   # holders / array_refs / key_str (importable: no guard, no I/O, no production)

SCHEMA = 'p515_s56_dependence_profiles_v1'
POINTS = ('checksum', 'post_read')
STATISTIC = ('ND[s] = sum_{loads l not ADN} sum_t pd_l[s,t] * baseMVA * dt - sum_{generators g, gen_type in '
             'definitions.GEN_CURTAILLABLE_TYPES} sum_t pg_g[s,t] * baseMVA * dt, in MWh, dt = 24 h / num_instants; '
             'per network block (network, year, day); ADN loads = the TSO loads production zeroes after the checksum '
             'point (network.adn_load_exists / get_adn_load_idx for every DSO node); evaluation: float(np.sum(A[s])) '
             '* baseMVA * dt per array, accumulated in Python floats, loads by index then generators by index')
TIE_RULE = ('perm = sorted(range(n), key=lambda s: (ND[s], s)): ascending ND, exact ties -> lower original index '
            'first; new index k takes original index perm[k]; new_A = old_A[perm, :] for every operation array of the '
            'block (market arrays never permuted)')
ARRAY_SHA_DEFINITION = ('sha256 over b"<f8" + repr(shape).encode("ascii") + np.ascontiguousarray(A, dtype="<f8")'
                        '.tobytes() (the array as float64, C order)')


class DependenceInstanceMismatch(RuntimeError):
    """Raised when a planning object's scenario arrays are not the declared dependence instance (or its base)."""

    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def array_sha(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(b'<f8')
    h.update(repr(tuple(a.shape)).encode('ascii'))
    h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def block_key(hname, year, day):
    return f'{hname}|{year}|{day}'


def block_of(key):
    """key tuple (from array_refs) -> block key string; None for market keys."""
    if key[0] == 'market':
        return None
    return block_key(key[0], key[1], key[2])


def comonotone_permutation(values):
    """Ascending values, exact ties -> lower original index first (TIE_RULE)."""
    vals = [float(v) for v in values]
    return sorted(range(len(vals)), key=lambda s: (vals[s], s))


def adn_load_indices(planning, net, hname):
    """Indices of the loads production zeroes as ADN loads (only in the TSO): for every DSO node n with
    net.adn_load_exists(n), net.get_adn_load_idx(n) -- production's own rule
    (shared_resources_planning._add_adn_node_to_transmission_network)."""
    if hname != 'TSO':
        return []
    out = []
    for node_id in planning.distribution_networks:
        if net.adn_load_exists(node_id):
            out.append(net.get_adn_load_idx(node_id))
    return sorted(set(out))


def daily_net_demand(planning, include_adn_loads=False):
    """{block key: {'values': [ND[s] for s in original order] (MWh), 'load_MWh': [...], 'res_MWh': [...],
    'adn_load_indices': [...], 'res_generators': [[index, gen_id, gen_type], ...], 'n_scenarios': n}} over every network
    block of the TSO and every DSO. See STATISTIC."""
    from definitions import GEN_CURTAILLABLE_TYPES   # production constant; imported lazily
    out = {}
    for hname, holder in EVI.holders(planning):
        n = int(holder.num_oper_scenarios)
        for year in sorted(holder.years):
            for day in holder.days:
                net = holder.network[year][day]
                ni = int(net.num_instants)
                if 24 % ni != 0:
                    raise DependenceInstanceMismatch(f'{hname}|{year}|{day}: num_instants {ni} does not divide 24', {})
                dt = 24.0 / ni
                base = float(net.baseMVA)
                adn = adn_load_indices(planning, net, hname)
                load_e = [0.0] * n
                res_e = [0.0] * n
                for i, load in enumerate(net.loads):
                    if i in adn and not include_adn_loads:
                        continue
                    A = np.asarray(load.pd, dtype=float)
                    if A.shape != (n, ni):
                        raise DependenceInstanceMismatch(f'{hname}|{year}|{day} load {i}: pd shape {A.shape} != '
                                                         f'{(n, ni)}', {})
                    for s in range(n):
                        load_e[s] += float(np.sum(A[s])) * base * dt
                res_gens = []
                for g, gen in enumerate(net.generators):
                    if gen.gen_type not in GEN_CURTAILLABLE_TYPES:
                        continue
                    res_gens.append([g, str(gen.gen_id), int(gen.gen_type)])
                    A = np.asarray(gen.pg, dtype=float)
                    if A.shape != (n, ni):
                        raise DependenceInstanceMismatch(f'{hname}|{year}|{day} gen {g}: pg shape {A.shape} != '
                                                         f'{(n, ni)}', {})
                    for s in range(n):
                        res_e[s] += float(np.sum(A[s])) * base * dt
                out[block_key(hname, year, day)] = {
                    'values': [load_e[s] - res_e[s] for s in range(n)], 'load_MWh': load_e, 'res_MWh': res_e,
                    'adn_load_indices': adn, 'res_generators': res_gens, 'n_scenarios': n, 'dt_h': dt}
    return out


# ======================================================================================================================
#  the derivation
# ======================================================================================================================
def derive(planning):
    """The comonotone re-pairing of `planning`'s operation scenarios, from the arrays as they are now (does not modify
    `planning`). Returns
      {'statistic': daily_net_demand(planning) (original order),
       'permutations': {block key: perm (list of original indices, new index k <- perm[k])},
       'identity': {block key: bool},
       'keys': [operation key strings in array_refs order],
       'permuted': {key string: new ndarray (n x T) = A[perm, :]},
       'base_sha256': {key: array_sha(A)}, 'permuted_sha256': {key: array_sha(A[perm])},
       'market_sha256': {market key: array_sha(A)}}"""
    stat = daily_net_demand(planning)
    perms = {bk: comonotone_permutation(v['values']) for bk, v in stat.items()}
    out = {'statistic': stat, 'permutations': perms,
           'identity': {bk: p == list(range(len(p))) for bk, p in perms.items()},
           'keys': [], 'permuted': {}, 'base_sha256': {}, 'permuted_sha256': {}, 'market_sha256': {}}
    for key, kind, arr, _w, _b in EVI.array_refs(planning):
        ks = EVI.key_str(key)
        if kind == 'market':
            out['market_sha256'][ks] = array_sha(arr)
            continue
        bk = block_of(key)
        perm = perms[bk]
        A = np.asarray(arr)
        if A.ndim != 2 or A.shape[0] != len(perm) or A.dtype != np.float64:
            raise DependenceInstanceMismatch(f'derive {ks}: array {A.dtype} {A.shape} does not carry {len(perm)} '
                                             f'operation scenarios', {'key': ks})
        new = A[perm, :].copy()
        out['keys'].append(ks)
        out['permuted'][ks] = new
        out['base_sha256'][ks] = array_sha(A)
        out['permuted_sha256'][ks] = array_sha(new)
    return out


# ======================================================================================================================
#  the profiles file
# ======================================================================================================================
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_profiles(path, expected_sha256=None):
    """Read a dependence profiles file (schema SCHEMA). Raises on a schema or sha256 mismatch or inconsistent key
    sets. Returns the payload dict with 'path' and 'sha256' added."""
    sha = _sha256_file(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise DependenceInstanceMismatch(f'profiles file {path}: sha256 {sha} != expected {expected_sha256}',
                                         {'profiles_sha256': sha, 'expected_sha256': expected_sha256})
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get('schema') != SCHEMA:
        raise DependenceInstanceMismatch(f'profiles file {path}: schema {payload.get("schema")!r} != {SCHEMA!r}',
                                         {'profiles_sha256': sha})
    keys = payload['operation_keys']
    bad = (set(payload['base_sha256']) != set(keys) or set(payload['permuted_sha256']) != set(keys)
           or not set(payload['post_read_expected_sha256']) <= set(keys)
           or set(payload['permutations']) != set(payload['statistic_original_order']))
    if bad:
        raise DependenceInstanceMismatch(f'profiles file {path}: inconsistent key sets', {'profiles_sha256': sha})
    payload = dict(payload)
    payload['path'] = path
    payload['sha256'] = sha
    return payload


# ======================================================================================================================
#  the injection (permutation) hook
# ======================================================================================================================
def _check_dims_and_probabilities(planning, profiles):
    fails = []
    n = int(profiles['n_operation_scenarios'])
    if int(getattr(planning, 'num_market_scenarios', -1)) != int(profiles['n_market_scenarios']):
        fails.append(f'num_market_scenarios {getattr(planning, "num_market_scenarios", None)} != '
                     f'{profiles["n_market_scenarios"]}')
    for hname, holder in EVI.holders(planning):
        if int(holder.num_oper_scenarios) != n:
            fails.append(f'{hname}: num_oper_scenarios {holder.num_oper_scenarios} != {n}')
    for y in planning.years:
        if [float(v) for v in planning.prob_market_scenarios[y]] != profiles['market_probabilities'][str(y)]:
            fails.append(f'planning.prob_market_scenarios[{y}] != stored')
    bad = []
    for hname, holder in EVI.holders(planning):
        for y in holder.years:
            for d in holder.days:
                net = holder.network[y][d]
                bk = block_key(hname, y, d)
                exp = profiles['block_probabilities'].get(bk)
                got = {'operation': [float(v) for v in net.prob_operation_scenarios],
                       'market': [float(v) for v in net.prob_market_scenarios]}
                if exp != got:
                    bad.append(bk)
    if bad:
        fails.append(f'{len(bad)} network blocks with probability vectors != stored (first {bad[:3]})')
    return fails


def apply(planning, profiles):
    """At the checksum point: verify the live arrays are the base instance (dimensions, probabilities, key sets, every
    operation and market array's sha256 == stored base), recompute the permutation from them (`derive`) and require it
    equal to the stored one, then write A[perm, :] IN PLACE into every operation array (checked against the stored
    permuted sha256). Market arrays are not touched. Returns counts."""
    fails = _check_dims_and_probabilities(planning, profiles)
    if fails:
        raise DependenceInstanceMismatch('dependence hook: ' + ' | '.join(fails), {'failures': fails})
    refs = EVI.array_refs(planning)
    op = [(EVI.key_str(k), arr) for k, kind, arr, _w, _b in refs if kind == 'operation']
    mk = [(EVI.key_str(k), arr) for k, kind, arr, _w, _b in refs if kind == 'market']
    if [k for k, _ in op] != list(profiles['operation_keys']) or {k for k, _ in mk} != set(profiles['market_sha256']):
        raise DependenceInstanceMismatch('dependence hook: the read\'s key list differs from the stored one', {})
    if len({id(a) for _k, a in op}) != len(op):
        raise DependenceInstanceMismatch('dependence hook: two operation keys share one ndarray (aliasing)', {})
    bad_base = [k for k, a in op if array_sha(a) != profiles['base_sha256'][k]]
    bad_mk = [k for k, a in mk if array_sha(a) != profiles['market_sha256'][k]]
    if bad_base or bad_mk:
        raise DependenceInstanceMismatch(
            f'dependence hook: the read is not the base instance: {len(bad_base)} operation and {len(bad_mk)} market '
            f'arrays differ (first {(bad_base + bad_mk)[:3]})', {'n_bad_base': len(bad_base), 'n_bad_market': len(bad_mk)})
    der = derive(planning)
    if der['permutations'] != profiles['permutations']:
        diff = [bk for bk in profiles['permutations'] if der['permutations'].get(bk) != profiles['permutations'][bk]]
        raise DependenceInstanceMismatch(f'dependence hook: recomputed permutations differ in {len(diff)} blocks '
                                         f'(first {diff[:3]})', {'blocks': diff[:20]})
    stat_bad = [bk for bk, v in der['statistic'].items() if v['values'] != profiles['statistic_original_order'][bk]]
    if stat_bad:
        raise DependenceInstanceMismatch(f'dependence hook: recomputed statistic differs in {len(stat_bad)} blocks',
                                         {'blocks': stat_bad[:20]})
    n_changed = 0
    for k, a in op:
        new = der['permuted'][k]
        if array_sha(new) != profiles['permuted_sha256'][k]:
            raise DependenceInstanceMismatch(f'dependence hook {k}: permuted array sha256 != stored', {'key': k})
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != new.shape:
            raise DependenceInstanceMismatch(f'dependence hook {k}: live array cannot take the permuted array', {'key': k})
        if profiles['permuted_sha256'][k] != profiles['base_sha256'][k]:
            n_changed += 1
        a[...] = new
    return {'n_operation_arrays_permuted': len(op), 'n_arrays_changed': n_changed,
            'n_market_arrays_checked_untouched': len(mk),
            'n_blocks_non_identity': sum(1 for v in der['identity'].values() if not v)}


class DependenceHook:
    """Pass-through wrapper on shared_resources_planning._compute_scenario_metadata (the point the EV twin's
    p515_s56_ev_injection.ScenarioPointHook uses): `apply(planning, profiles)`, then `on_point(planning)` if given, then
    the original (the scenario checksum is computed over the permuted arrays). profiles=None: capture only."""

    def __init__(self, profiles=None, on_point=None):
        import shared_resources_planning as SRP
        self._srp = SRP
        self.original = SRP._compute_scenario_metadata
        self.profiles = profiles
        self.on_point = on_point
        self.calls = 0
        self.apply_counts = []
        self.installed = False

    def install(self):
        hook = self

        def wrapped(planning_problem):
            hook.calls += 1
            if hook.profiles is not None:
                hook.apply_counts.append(apply(planning_problem, hook.profiles))
            if hook.on_point is not None:
                hook.on_point(planning_problem)
            return hook.original(planning_problem)
        self._srp._compute_scenario_metadata = wrapped
        self.installed = True
        return self

    def uninstall(self):
        self._srp._compute_scenario_metadata = self.original
        self.installed = False


# ======================================================================================================================
#  the in-child assertion
# ======================================================================================================================
def assert_dependence_arrays(planning, profiles, point='post_read', expected_scenario_checksum=None, max_report=20):
    """RAISES DependenceInstanceMismatch unless, in `planning` at `point` ('checksum' after the hook, or 'post_read'):
      - scenario dimensions and every probability vector equal the stored ones (1/3 each in the S56 instances);
      - the operation / market key lists are the stored ones;
      - every operation array's sha256 == the stored PERMUTED sha256 (at 'post_read' the TSO ADN pd / qd are compared
        with 'post_read_expected_sha256');
      - NOT THE INDEX-PAIRED INSTANCE: the number of operation arrays whose sha256 differs from the stored base sha256
        equals the stored count of changed arrays at that point (> 0), and no array that the permutation changes
        still carries its base bytes;
      - every market array (and every network block's own cost_energy_p / cost_flex alias) == the stored sha256
        (unchanged by construction);
      - the statistic recomputed from the live arrays is non-decreasing in the scenario index in every block, and the
        comonotone permutation of the live arrays is the identity (the run IS the re-paired instance);
      - if `expected_scenario_checksum` is given: planning.scenario_metadata['combined_scenario_checksum'] equals it,
        and it differs from the stored base checksum.
    Returns the summary dict when everything holds."""
    if point not in POINTS:
        raise ValueError(f'point must be one of {POINTS}, got {point!r}')
    failures = _check_dims_and_probabilities(planning, profiles)
    refs = EVI.array_refs(planning)
    op = [(EVI.key_str(k), arr) for k, kind, arr, _w, _b in refs if kind == 'operation']
    mk = [(EVI.key_str(k), arr) for k, kind, arr, _w, _b in refs if kind == 'market']
    if [k for k, _ in op] != list(profiles['operation_keys']) or {k for k, _ in mk} != set(profiles['market_sha256']):
        failures.append(f'key lists differ: read {len(op)} operation / {len(mk)} market, stored '
                        f'{len(profiles["operation_keys"])} / {len(profiles["market_sha256"])}')
    post = profiles['post_read_expected_sha256'] if point == 'post_read' else {}
    mism, still_base, n_diff_from_base = [], [], 0
    for k, a in op:
        if k not in profiles['permuted_sha256']:
            continue
        got = array_sha(a)
        exp = post.get(k, profiles['permuted_sha256'][k])
        base = post.get(k, profiles['base_sha256'][k])
        if got != exp:
            mism.append({'key': k, 'what': 'sha256 != stored permuted', 'equals_base': got == base})
        if got != base:
            n_diff_from_base += 1
        elif exp != base:
            still_base.append(k)
    exp_changed = profiles['n_arrays_changed'][point]
    if mism:
        failures.append(f'{len(mism)} of {len(op)} operation arrays differ from the stored permuted arrays '
                        f'(first {mism[0]}; {sum(1 for m in mism if m["equals_base"])} of them carry the BASE bytes)')
    if still_base or n_diff_from_base != exp_changed or exp_changed <= 0:
        failures.append(f'index-paired check: {n_diff_from_base} operation arrays differ from the base instance '
                        f'(expected {exp_changed} > 0); {len(still_base)} arrays the permutation changes still carry '
                        f'their base bytes (first {still_base[:3]})')
    mk_bad = [k for k, a in mk if array_sha(a) != profiles['market_sha256'][k]]
    if mk_bad:
        failures.append(f'{len(mk_bad)} market arrays differ from the stored (unchanged) ones (first {mk_bad[:3]})')
    alias_bad = []
    for hname, holder in EVI.holders(planning):
        for y in holder.years:
            for d in holder.days:
                net = holder.network[y][d]
                for fld, tag in (('cost_energy_p', 'energy'), ('cost_flex', 'flex')):
                    ks = EVI.key_str(('market', tag, y, d))
                    if ks not in profiles['market_sha256'] or array_sha(getattr(net, fld)) != profiles['market_sha256'][ks]:
                        alias_bad.append(f'{hname}|{y}|{d}|{fld}')
    if alias_bad:
        failures.append(f'{len(alias_bad)} network-block market arrays differ (first {alias_bad[:3]})')
    stat = daily_net_demand(planning)
    non_mono = [bk for bk, v in stat.items() if any(v['values'][s + 1] < v['values'][s]
                                                     for s in range(len(v['values']) - 1))]
    non_identity = [bk for bk, v in stat.items() if comonotone_permutation(v['values']) != list(range(len(v['values'])))]
    if non_mono or non_identity:
        failures.append(f'statistic not non-decreasing in {len(non_mono)} blocks / comonotone permutation not the '
                        f'identity in {len(non_identity)} blocks (first {(non_mono + non_identity)[:3]})')
    got_ck = (getattr(planning, 'scenario_metadata', None) or {}).get('combined_scenario_checksum')
    if expected_scenario_checksum is not None:
        if got_ck != expected_scenario_checksum:
            failures.append(f'combined scenario checksum {got_ck} != expected {expected_scenario_checksum}')
        if expected_scenario_checksum == profiles['base_instance']['scenario_checksum']:
            failures.append('the expected checksum is the BASE (index-paired) instance\'s')
    summary = {'point': point, 'profiles_sha256': profiles.get('sha256'), 'n_operation_arrays': len(op),
               'n_market_arrays': len(mk), 'n_mismatches': len(mism), 'mismatches_first': mism[:max_report],
               'n_differ_from_base': n_diff_from_base, 'expected_n_differ_from_base': exp_changed,
               'n_still_base': len(still_base), 'n_market_bad': len(mk_bad), 'n_alias_bad': len(alias_bad),
               'n_blocks_non_monotone': len(non_mono), 'n_blocks_non_identity_now': len(non_identity),
               'combined_scenario_checksum': got_ck, 'expected_scenario_checksum': expected_scenario_checksum,
               'failures': failures, 'ok': not failures}
    if failures:
        raise DependenceInstanceMismatch('dependence-instance assertion FAILED: ' + ' | '.join(failures), summary)
    return summary
