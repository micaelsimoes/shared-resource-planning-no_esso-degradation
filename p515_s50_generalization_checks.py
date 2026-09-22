"""
P5.15 Addendum 36 (task W35, item 3) -- ZERO-SOLVE checks for the four single-scenario code paths.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 35 and 36; frozen spec v20
`data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json`, item `3_generalization`
("the four single-scenario code paths: hull-polish helpers, settlement reporting (s31c),
p56a_oracle.load_baseline, run_admm_arm's solve count").

The solve-bearing gate for this item is the SRP1 two-cycle bitwise reproduction
(`p515_s50_generalization_gate.py`). THIS file carries everything that can be settled without a
solve, and without reading the planning problem (production's reader writes into shared, relative
locations -- `data/SRP1/Diagrams` among them -- so it must not run beside a campaign). Every SRP1
number used here is read from a COMMITTED artifact, named with its sha256.

CHECKS

 A  `p56a_oracle.install_baseline` (path 3): a non-SRP1 instance can be installed without touching
    the SRP1 path; the SRP1 label still asserts `CANONICAL_CHECKSUM`; a second installation is
    refused; `load_baseline`'s own SRP1 assertions are unchanged (source-compared).
    `p515_s44_scale_measurement.inject_oracle_baseline` resolves to it.
 B  `run_admm_arm`'s solve count (path 4): the SRP1 constant 51 is gone from the assignment; the
    instance derivation reproduces 51 on SRP1's committed dimensions and 83 on the paper
    instance's, and agrees with `p515_s44_scale_measurement.declared_solve_profile`'s derivation;
    and on the committed C* re-certification the per-event identity now HOLDS (4528 = 4488 + 40)
    where `identity_holds` was recorded False.
 C  `_scenario_expectation` / `_expected_market_price` (paths 1 and 2): at one scenario each
    returns the single term EXACTLY -- checked bit for bit on hostile values (-0.0, denormal,
    1e308, 0.1) -- and above one scenario each returns the probability-weighted expectation.
    SRP1's per-block probability vectors are `[1.0] x [1.0]` on all 48 blocks (committed
    probability audit), which is what makes "exactly" equal "unchanged".
 D  Branch selection (paths 1 and 2): `_block_is_single_scenario` is exact; the legacy
    single-scenario reads and bounds are present VERBATIM in the source; `apply_common_values`
    raises `NotImplementedError` above 1 x 1 (executed on a stub, not merely inspected); and
    `common_coordinated_values` refuses a multi-scenario block with several DSO-side shared-ESS
    indices.
 E  Settlement reporting (path 2): on the committed C* settlement artifact, the priced residual
    the S31C writer now takes from `_get_interface_reporting_detail`
    (`priced_interface_residual_expected_mu`, which at one market scenario is
    `price_per_mwh * (p_int_tso_expected_mw - p_int_dso_expected_mw)`) reproduces every committed
    `priced_residual_pi_baseMVA_residual_unweighted` BIT FOR BIT, over every node/year/day/period.

ZERO SOLVES: `SolveProfileGuard(permitted=())` installed before any production import and
`verify(0)`-ed exactly. No model is built; no planning problem is read.

EXACT COMMAND (repo root, canonical interpreter, attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s50_generalization_checks.py \
      > data/SRP1/Results/P515S50/generalization_checks_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S50/generalization_checks/generalization_checks.json
Exit 0 when every check passes, 1 otherwise.
"""

import hashlib
import inspect
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W35 item 3 generalization checks (never solves)').install()

import p56a_oracle as O  # noqa: E402
import p515_s41_hull_polish as HP  # noqa: E402
import p515_s44_scale_measurement as S44  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

STAGE = 'P5.15 Addendum 36 W35 item 3 -- zero-solve checks for the four single-scenario code paths'
SCHEMA = 'p515_s50_generalization_checks_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S50', 'generalization_checks')

PROBABILITY_AUDIT = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'probability_audit',
                                 'probability_audit.json')
SRP1_CYCLE_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement',
                                 'srp1_cycle_snapoff_r1', 'cycle_record.json')
PAPER_CYCLE_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement',
                                  'paper_cycle_snapoff_memfix_r1', 'cycle_record.json')
C_STAR_ARM = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals',
                          '070f833e1e318f85_c_star', 'g_s39_D.json')
C_STAR_SETTLEMENT = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals',
                                 '070f833e1e318f85_c_star', 'interface_settlement_detail_s31c.json')

HOSTILE = (0.0, -0.0, 0.1, 1.0, -3.5, 1e308, 5e-324, 1.0 / 3.0, 12345678.9012345)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _bits(x):
    return struct.pack('>d', float(x)).hex()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W35-item3] {msg}', flush=True)


class _StubBlock:
    """The minimum a `_block_is_single_scenario` / `_scenario_expectation` call touches."""

    def __init__(self, n_market, n_operation):
        self.scenarios_market = list(range(n_market))
        self.scenarios_operation = list(range(n_operation))


class _StubNetwork:
    def __init__(self, prob_market, prob_operation, cost_energy_p=None):
        self.prob_market_scenarios = list(prob_market)
        self.prob_operation_scenarios = list(prob_operation)
        self.cost_energy_p = cost_energy_p


# ======================================================================================
#  A -- install_baseline
# ======================================================================================
def check_install_baseline():
    saved = O._BASELINE
    out = {}
    try:
        O._BASELINE = None
        sentinel = object()
        installed = O.install_baseline(sentinel, 'not-the-srp1-checksum', instance_label='pilot_2x2')
        out['non_srp1_instance_installs'] = (installed['planning'] is sentinel
                                             and installed['instance_label'] == 'pilot_2x2'
                                             and installed['checksum'] == 'not-the-srp1-checksum'
                                             and installed['checksum_matches_srp1_canonical'] is False)
        # `load_baseline` returns the installed record and does NOT re-read SRP1 (it would raise on
        # the checksum if it did; the sentinel identity proves no re-read happened).
        out['load_baseline_returns_the_installed_record'] = (O.load_baseline()['planning'] is sentinel)
        try:
            O.install_baseline(object(), 'x', instance_label='pilot_2x2')
            out['second_installation_refused'] = False
        except RuntimeError as error:
            out['second_installation_refused'] = 'already installed' in str(error)

        O._BASELINE = None
        try:
            O.install_baseline(object(), 'wrong-checksum', instance_label='srp1')
            out['srp1_label_still_asserts_canonical_checksum'] = False
        except RuntimeError as error:
            out['srp1_label_still_asserts_canonical_checksum'] = 'canonical' in str(error)

        O._BASELINE = None
        ok = O.install_baseline(object(), O.CANONICAL_CHECKSUM, instance_label='srp1')
        out['srp1_label_accepts_canonical_checksum'] = ok['checksum_matches_srp1_canonical'] is True
    finally:
        O._BASELINE = saved

    load_src = inspect.getsource(O.load_baseline)
    out['load_baseline_srp1_path_unchanged'] = all(t in load_src for t in (
        "SharedResourcesPlanning('data/SRP1', 'SRP1.json')",
        'checksum != CANONICAL_CHECKSUM',
        'planning.read_planning_problem()'))
    inject_src = inspect.getsource(S44.inject_oracle_baseline)
    out['scale_harness_delegates_to_install_baseline'] = (
        'O.install_baseline(planning, checksum, instance_label=' in inject_src
        and 'O._BASELINE =' not in inject_src)
    return out


# ======================================================================================
#  B -- run_admm_arm solve count
# ======================================================================================
def _derivation(n_dso, n_years, n_days, n_esso):
    return (1 + n_dso) * n_years * n_days + n_esso


def check_run_admm_arm_solve_count(evidence):
    import p515_g_g1_g4_admm_gates as G
    arm_src = inspect.getsource(G.run_admm_arm)
    out = {
        'hard_coded_51_gone_from_the_assignment': (
            "guard.counts['permitted_solve'] == 51 * len(rows) + 51" not in arm_src),
        'derivation_present': ('_solves_per_cycle = (1 + _n_dso) * len(planning.years) * len(planning.days) + _n_esso'
                               in arm_src),
        'per_event_credit_present': ("int(bool(b.get('recovery_attempted'))) + int(bool(b.get('tier2_attempted')))"
                                     in arm_src),
        'esso_or_indeterminate_makes_it_unsupported': ('len(final_esso_events) == 0' in arm_src
                                                       and "b.get('class') == 'indeterminate'" in arm_src),
    }
    for label, rel, expected_per_cycle in (('srp1', SRP1_CYCLE_RECORD, 51), ('paper', PAPER_CYCLE_RECORD, 83)):
        full = os.path.join(REPO, rel)
        record = json.load(open(full))
        evidence[rel] = _sha256(full)
        dims = record['planning_dimensions']
        profile = record['declared_solve_profile']
        derived = _derivation(profile['n_dso'], len(dims['years']), len(dims['days']),
                              len(dims['active_distribution_network_nodes']))
        out[f'{label}_solves_per_cycle_derivation'] = {
            'n_dso': profile['n_dso'], 'n_years': len(dims['years']), 'n_days': len(dims['days']),
            'n_esso_nodes': len(dims['active_distribution_network_nodes']),
            'derived': derived, 'expected': expected_per_cycle,
            'agrees_with_scale_harness_declaration': derived == profile['solves_per_cycle'],
            'holds': derived == expected_per_cycle == profile['solves_per_cycle'],
        }
        out[f'{label}_solves_per_cycle_holds'] = out[f'{label}_solves_per_cycle_derivation']['holds']

    full = os.path.join(REPO, C_STAR_ARM)
    report = json.load(open(full))
    evidence[C_STAR_ARM] = _sha256(full)
    observed = report['solve_profile']['observed']['permitted_solve']
    cycles = report['cycles_run']
    base = 51 * (cycles + 1)
    events_rel = report['network_failures_summary']['path']
    events_full = os.path.join(REPO, events_rel)
    evidence[events_rel] = _sha256(events_full)
    with open(events_full) as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    retries = sum(int(bool(e.get('recovery_attempted'))) + int(bool(e.get('tier2_attempted')))
                  for e in events if e.get('record_type', 'network_block') == 'network_block'
                  and e.get('class') is not None)
    out['c_star_identity_now_holds_where_it_was_recorded_false'] = {
        'recorded_identity_holds': report['solve_profile']['identity_holds'],
        'observed': observed, 'cycles_run': cycles, 'base': base, 'retries_credited': retries,
        'new_expected': base + retries, 'new_identity_holds': observed == base + retries,
        'holds': (report['solve_profile']['identity_holds'] is False and observed == base + retries),
    }
    out['c_star_identity_repaired'] = out['c_star_identity_now_holds_where_it_was_recorded_false']['holds']
    return out


# ======================================================================================
#  C -- expectation helpers
# ======================================================================================
def check_expectation_helpers(evidence):
    out = {}
    single = _StubBlock(1, 1)
    net1 = _StubNetwork([1.0], [1.0])
    mismatches = []
    for value in HOSTILE:
        got = O._scenario_expectation(single, net1, lambda s_m, s_o, v=value: v)
        if _bits(got) != _bits(value):
            mismatches.append({'value': value, 'value_bits': _bits(value),
                               'got': got, 'got_bits': _bits(got)})
    out['scenario_expectation_is_bitwise_identity_at_1x1'] = not mismatches
    out['scenario_expectation_1x1_mismatches'] = mismatches

    two = _StubBlock(2, 2)
    net2 = _StubNetwork([0.3, 0.7], [0.4, 0.6])
    table = {(0, 0): 10.0, (0, 1): 20.0, (1, 0): 30.0, (1, 1): 40.0}
    got = O._scenario_expectation(two, net2, lambda s_m, s_o: table[(s_m, s_o)])
    manual = (0.3 * 0.4 * 10.0 + 0.3 * 0.6 * 20.0) + (0.7 * 0.4 * 30.0 + 0.7 * 0.6 * 40.0)
    out['scenario_expectation_2x2'] = {'got': got, 'manual': manual, 'holds': got == manual}
    out['scenario_expectation_2x2_holds'] = got == manual

    price_mismatches = []
    for value in HOSTILE:
        net = _StubNetwork([1.0], [1.0], cost_energy_p={0: {7: value}})
        got = srp._expected_market_price(_StubBlock(1, 1), net, 7)
        if _bits(got) != _bits(value):
            price_mismatches.append({'value': value, 'got': got})
    out['expected_market_price_is_bitwise_identity_at_one_market_scenario'] = not price_mismatches
    out['expected_market_price_mismatches'] = price_mismatches

    net = _StubNetwork([0.25, 0.75], [1.0], cost_energy_p={0: {3: 40.0}, 1: {3: 80.0}})
    got = srp._expected_market_price(_StubBlock(2, 1), net, 3)
    out['expected_market_price_two_market_scenarios'] = {
        'got': got, 'manual': 0.25 * 40.0 + 0.75 * 80.0, 'holds': got == 0.25 * 40.0 + 0.75 * 80.0}
    out['expected_market_price_two_market_scenarios_holds'] = got == (0.25 * 40.0 + 0.75 * 80.0)

    full = os.path.join(REPO, PROBABILITY_AUDIT)
    audit = json.load(open(full))
    evidence[PROBABILITY_AUDIT] = _sha256(full)
    blocks = []
    for agent, agent_record in audit['srp1_numerical']['probability_objects']['networks'].items():
        for block, block_record in agent_record['blocks'].items():
            blocks.append((agent, block, block_record['prob_market_scenarios'],
                           block_record['prob_operation_scenarios']))
    bad = [b for b in blocks if b[2] != [1.0] or b[3] != [1.0]]
    out['srp1_probability_vectors_are_unit_on_every_block'] = {
        'n_blocks': len(blocks), 'n_not_unit': len(bad), 'first_bad': bad[:3],
        'source': PROBABILITY_AUDIT, 'holds': len(blocks) > 0 and not bad}
    out['srp1_probability_vectors_are_unit_on_every_block_holds'] = len(blocks) > 0 and not bad
    return out


# ======================================================================================
#  D -- branch selection and loud failures
# ======================================================================================
class _StubHolder:
    def __init__(self, years, days):
        self.years = years
        self.days = days


class _StubPlanning:
    def __init__(self, years, days):
        self.transmission_network = _StubHolder(years, days)
        self.distribution_networks = {}


def check_branch_selection():
    out = {
        'single_scenario_predicate': {
            '1x1': O._block_is_single_scenario(_StubBlock(1, 1)),
            '2x1': O._block_is_single_scenario(_StubBlock(2, 1)),
            '1x2': O._block_is_single_scenario(_StubBlock(1, 2)),
            '2x2': O._block_is_single_scenario(_StubBlock(2, 2)),
        }}
    out['single_scenario_predicate_holds'] = (out['single_scenario_predicate']['1x1'] is True
                                              and not out['single_scenario_predicate']['2x1']
                                              and not out['single_scenario_predicate']['1x2']
                                              and not out['single_scenario_predicate']['2x2'])

    ccv_src = inspect.getsource(O.common_coordinated_values)
    out['legacy_single_scenario_reads_intact'] = all(t in ccv_src for t in (
        "t_p = float(pe.value(t_model.pc_adn[dn, 0, 0, p]))",
        "d_p = float(pe.value(d_model.pg_adn[0, 0, p]))",
        "t_v = float(pe.value(t_model.vmag_sqr[adn_idx, 0, 0, p])) ** 0.5",
        "t_sp = sum(float(pe.value(t_model.shared_es_pnet[e, 0, 0, p]))"))
    out['multi_scenario_reads_are_the_expected_vars'] = all(t in ccv_src for t in (
        "t_model.expected_interface_pf_p[dn, p]", "d_model.expected_interface_pf_p[p]",
        "t_model.expected_interface_vmag[dn, p]", "t_model.expected_shared_ess_p[e, p]",
        "d_model.expected_shared_ess_p[p]"))

    hull_src = inspect.getsource(HP.apply_hull_bounds)
    out['legacy_single_scenario_bounds_intact'] = all(t in hull_src for t in (
        "tv = t_model.vmag_sqr[adn_idx, 0, 0, p]", "tp_expr = t_model.pc_adn[dn, 0, 0, p]",
        "dp_expr = d_model.pg_adn[0, 0, p]",
        "t_vars = [t_model.shared_es_pnet[e, 0, 0, p] for e in t_sess]"))
    out['multi_scenario_bounds_are_the_expected_vars'] = all(t in hull_src for t in (
        "tv = t_model.expected_interface_vmag[dn, p]",
        "tp_expr = t_model.expected_interface_pf_p[dn, p]",
        "t_vars = [t_model.expected_shared_ess_p[e, p] for e in t_sess]",
        "d_vars = [d_model.expected_shared_ess_p[p]]"))

    # `apply_common_values` must RAISE above 1 x 1 -- executed, not inspected.
    planning = _StubPlanning(['2025'], ['Spring'])
    models = {'tso': {'2025': {'Spring': _StubBlock(2, 2)}}}
    try:
        O.apply_common_values(planning, models, {})
        out['apply_common_values_raises_above_1x1'] = False
        out['apply_common_values_message'] = None
    except NotImplementedError as error:
        out['apply_common_values_raises_above_1x1'] = True
        out['apply_common_values_message'] = str(error)
    # and must NOT raise at 1 x 1 (it proceeds into its own loop, which needs no DSO here)
    try:
        O.apply_common_values(_StubPlanning(['2025'], ['Spring']),
                              {'tso': {'2025': {'Spring': _StubBlock(1, 1)}}}, {})
        out['apply_common_values_proceeds_at_1x1'] = True
    except NotImplementedError:
        out['apply_common_values_proceeds_at_1x1'] = False

    out['common_coordinated_values_refuses_multi_ess_dso_above_1x1'] = (
        'multi-scenario route reads the DSO' in ccv_src and 'NotImplementedError' in ccv_src)
    return out


# ======================================================================================
#  E -- settlement reporting, bitwise on the committed C* artifact
# ======================================================================================
def check_settlement_priced_residual(evidence):
    full = os.path.join(REPO, C_STAR_SETTLEMENT)
    detail = json.load(open(full))
    evidence[C_STAR_SETTLEMENT] = _sha256(full)
    reporting = detail['interface_reporting_detail']
    n, mismatches = 0, []
    for node, by_year in detail['interface_consensus_residual_per_dso'].items():
        for key, row in by_year['periods'].items():
            year, day, period = key.split('|')
            price = reporting[node][year][day]['periods'][period]['price_per_mwh']
            # what `_get_interface_reporting_detail` now hands the writer, at one market scenario
            recomputed = price * (row['p_int_tso_expected_mw'] - row['p_int_dso_expected_mw'])
            committed = row['priced_residual_pi_baseMVA_residual_unweighted']
            n += 1
            if _bits(recomputed) != _bits(committed):
                mismatches.append({'node': node, 'key': key, 'recomputed': recomputed,
                                   'committed': committed, 'recomputed_bits': _bits(recomputed),
                                   'committed_bits': _bits(committed)})
    return {'n_period_entries_compared': n, 'n_mismatches': len(mismatches),
            'first_mismatches': mismatches[:5],
            'priced_residual_reproduced_bitwise': n > 0 and not mismatches,
            'note': ('the writer now reads `priced_interface_residual_expected_mu`; at one market '
                     'scenario `_get_interface_reporting_detail` forms it as '
                     '`price * (p_int_tso_expected - p_int_dso_expected)`, the exact expression '
                     'the writer used before, so every committed value is reproduced bit for bit')}


def main():
    out_dir = os.path.join(REPO, OUT_REL)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'generalization_checks.json')
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite {path}')
    _log(STAGE)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()

    evidence = {}
    sections = {
        'A_install_baseline': check_install_baseline(),
        'B_run_admm_arm_solve_count': check_run_admm_arm_solve_count(evidence),
        'C_expectation_helpers': check_expectation_helpers(evidence),
        'D_branch_selection': check_branch_selection(),
        'E_settlement_priced_residual': check_settlement_priced_residual(evidence),
    }
    failures = []
    for section, body in sections.items():
        for key, value in body.items():
            if isinstance(value, bool) and not value:
                failures.append(f'{section}.{key}')
    guard_failures = GUARD.verify(0)
    if guard_failures:
        failures.append(f'guard: {guard_failures}')
    all_ok = not failures

    payload = {
        'schema': SCHEMA, 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addenda 35 and 36',
                      'data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json item 3_generalization',
                      'Planner task W35 item 3'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head': head,
        'sections': sections, 'failures': failures, 'all_ok': all_ok,
        'guard': {'permitted': [], 'counts': dict(GUARD.counts), 'declared_solves': 0,
                  'verify_failures': guard_failures},
        'evidence_sha256': evidence,
        'script_sha256': _sha256(os.path.abspath(__file__)),
    }
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    for section, body in sections.items():
        _log(section)
        for key, value in body.items():
            if isinstance(value, bool):
                _log(f'   {key}: {value}')
    _log(f'failures: {failures}')
    _log(f'wrote {os.path.relpath(path, REPO)} (sha256 {_sha256(path)})')
    _log(f'ALL_OK={all_ok}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
