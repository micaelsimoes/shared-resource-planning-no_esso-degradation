"""
P5.15 Addendum 36 -- W37: 2 x 2 multi-scenario SMOKE of the >1 x 1 hull-polish and
settlement branches.

================================================================================
WHAT THIS IS, AND WHAT IT IS NOT  (read this before reading any number below)
================================================================================
*** SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY.               ***
*** NOT THE PILOT INSTANCE. THE ROW-18 / alpha FORMULATION DECISION IS PENDING. ***

W36 established that the signed LINEAR row 18 is not implemented in this tree; what
exists above one scenario is the QUADRATIC scenario-deviation penalty
(`shared_resources_planning._add_tso_scenario_deviation_penalty` /
`_add_dso_scenario_deviation_penalty`, weights `definitions.PENALTY_SCENARIO_DEVIATION`
= 9e4 on voltage + interface power and `definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION`
= 1e4 on the shared ESS), added to `model.objective.expr` and therefore OUTSIDE the
`objective_function_rule` value Q(x) is read from. This stage runs the tree EXACTLY as
committed, with that quadratic active. Every number it produces is a BRANCH-EXECUTION
and SANITY observation for code that has never run above 1 x 1 -- never an economic
result, never a pilot measurement, never comparable with any SRP1 (1 x 1) figure.

The paths de-risked here are W35 item 3's generalizations (commit cb545653), none of
which had ever executed above 1 x 1:
  * `p56a_oracle.common_coordinated_values`      -- expectation-mode reads
  * `p56a_oracle.per_block_base_objectives`      -- expectation-weighted families
  * `p515_s41_hull_polish.hull_entries_with_esso` + `apply_hull_bounds`
                                                 -- expectation-coupled hull bounds
  * `shared_resources_planning._get_interface_reporting_detail` / `_expected_market_price`
    and the S31C writer `p515_g_g1_g4_admm_gates.write_interface_settlement_detail_s31c`
And the one path that MUST NOT be on them: `p56a_oracle.apply_common_values`, which
raises `NotImplementedError` above 1 x 1 by design. It is TRIPWIRED here (a counting
wrapper installed for the whole run), not asserted.

================================================================================
THE INSTANCE
================================================================================
Derived with `p515_s44_scale_measurement.derive_case` (the derived-case machinery of the
scale harness: it edits ONLY `Years`, `NumMarketScenarios` and every network's
`num_operation_scenarios` in a COPY of `data/SRP1/SRP1.json`, written into this stage's
write-once output directory; `data/SRP1/SRP1.json` itself is never touched):

    Years                    {2025: 5, 2030: 5, 2035: 5}  ->  {2025: 5}
    NumMarketScenarios       1                            ->  2
    num_operation_scenarios  1 (TSO + 3 DSOs)             ->  2 (TSO + 3 DSOs)
    Days                     UNCHANGED (Spring/Summer/Autumn/Winter)

so: 1 representative year x 4 days x 2 market x 2 operation scenarios
  = 4 scenario COMBINATIONS indexed inside each block,
  = (1 TSO + 3 DSO) x 1 year x 4 days = 16 network blocks + 3 ESSO models,
  = 19 solves per round.
The scenario realization is NEW (different year set and scenario counts give different
derived seeds), so its combined checksum necessarily differs from the SRP1 canonical one
and is NOT compared with it -- `p56a_oracle.install_baseline` is called with a non-SRP1
instance label, which is exactly the refusal-free route W35 item 3 added for this.

Ageing: the committed BASELINE (`data/SRP1/SharedESS/SRP1_ESS_Params.json` at 2466401d,
sha256 39106f93...), untouched. AA: ON from `data/SRP1/SRP1_params.json`, declared to
`p515_s44_campaign_harness._config_hook_factory` exactly as every other stage declares it.
Snapshots: OFF (declared, recorded, verified) -- a capture setting only; it removes the
pristine-clone build and tier-2 (frozen-snapshot) recovery, which is recorded in the
solve reconciliation.

================================================================================
ARMS AND SOLVE DECLARATION
================================================================================
Two arms, each a SHORT ADMM run (`p515_g_g1_g4_admm_gates.run_admm_arm`,
`num_max_iters_override = CYCLES` = 2, `apply_rho=False`, case-file rho). Certification is
NOT the point and is not reached:
    x0    -- x = 0 at every active node (5, 7, 9)
    unit  -- the smallest node-7 4 h unit, 0.25 MVA / 1.0 MWh at node 7 (the ladder's
             `UNIT1`, `p515_s50_marginal_campaign.UNIT1`), 0 at nodes 5 and 9
followed, on the arm's OWN final models, by the >1 x 1 paths (zero re-runs): the S31C
settlement detail + writer, `common_coordinated_values`, `hull_entries_with_esso`,
`per_block_base_objectives`, and finally `_polish_all_blocks_hull` (which applies the hull
bounds and re-solves every network block).

DECLARED, BEFORE THE RUN, and enforced by ONE bounded `SolveProfileGuard(PERMITTED)`
installed before any production import and `verify()`-ed EXACTLY:

    per arm  base   = solves_per_cycle x (1 initialization + CYCLES cycles)
                    = [(1 + n_dso) x n_years x n_days + n_esso] x (CYCLES + 1)
             polish = (1 + n_dso) x n_years x n_days          (one per network block)
    total    = sum over arms of (base + polish)

and the count GATED ON is that total plus the retries ACTUALLY ATTEMPTED, credited per
FAILURE EVENT by `p515_s44_scale_measurement.event_level_solve_reconciliation` (the
ladder's per-EVENT rule, W35 item 1(a)) for the ADMM part, plus the polish part's own
inner-guard surplus over one solve per block (production's tier-1/tier-2 recovery fires
inside `network.run_smopf` there too). Both retry terms are REPORTED separately; an
unsupported ADMM reconciliation (ESSO recovery event, 'indeterminate' event, event-file
mismatch) fails loudly. Too few solves fails as loudly as too many.

================================================================================
WHAT IS CHECKED  (each with its own numbers in the JSON and the Markdown)
================================================================================
 C1  every block is genuinely >1 x 1 and every generalized site takes the EXPECTATION
     branch (`expectation_mode` True, `source_agent_values` = the expected Vars).
 C2  every hull interval is FINITE, NON-EMPTY (lo <= hi), and is EXACTLY the closed hull
     of the agents' achieved values -- 2 agents (TSO, DSO) for V / PF_P / PF_Q and THREE
     (TSO, DSO, ESSO) for ESS_P / ESS_Q, per the Planner ruling: recomputed here
     independently with min/max and compared for bitwise equality with
     `p515_s41_hull_polish._interval`'s result as carried by `apply_hull_bounds`'s own
     descriptors. The count of entries where the ESSO endpoint is the BINDING end is
     reported, so "the ESSO endpoint is included" is a measured number, not a claim.
 C3  the polished blocks solve (per-block solved flags; the gate is not evaluated if any
     block fails -- `_polish_all_blocks_hull`'s own convention).
 C4  the per-block objective decomposition sums correctly. IDENTITY, stated here rather
     than in prose elsewhere:
         sum(families) == weighted_base_objective
                          - weight * interface_settlement_weight * interface_settlement
     because `model_construction_helpers.objective_function_rule` is the sum of the seven
     family aggregates PLUS the interface settlement term, and `per_block_base_objectives`
     decomposes only the seven. TOLERANCE: relative 1e-9 against
     max(|weighted_base_objective|, 1.0). Additionally, each family's expectation is
     checked against its own per-scenario values re-weighted by the network's probability
     vectors (relative 1e-9) -- this is what exercises `_scenario_expectation`'s
     multi-scenario branch.
 C5  the S31C priced residual reconciles as it does at 1 x 1. IDENTITY (the writer's own
     reading rule):
         t_tso_plus_t_dso_terminal == -1 x sum over DSO nodes of
                                      sum_pi_baseMVA_residual_weighted
     TOLERANCE: relative 1e-9 against max(|lhs|, |rhs|, 1.0). (At 1 x 1 the committed
     reference reproduced 840,010,674.369291 vs -840,010,674.369292, relative ~1.2e-15.)
 C6  `price_per_mwh_by_scenario` and the expectation-vs-per-scenario conventions are sane:
     every price finite; the per-scenario map carries n_m x n_o keys; prices depend on the
     MARKET index only; `price_per_mwh` equals sum_sm prob_market[s_m] * cost_energy_p
     [s_m][t] recomputed from the network object (relative 1e-12) and lies within the
     per-scenario range; the two market scenarios actually DIFFER (reported spread -- a
     zero spread would make the whole test vacuous); and the multi-market settlement
     branch is genuinely taken, evidenced by reporting `dso_settlement_sum_pi_p_int`
     beside the naive `sum_t E[pi_t] * E[p_int,t]` it deliberately is NOT.
 C7  nothing raises `NotImplementedError` except `p56a_oracle.apply_common_values`, which
     must NOT be on this path: a counting tripwire wraps it for the whole run and must
     record ZERO calls; one DELIBERATE probe call is then made, LAST, on the final models,
     and must raise `NotImplementedError` (the probe is flagged, counted separately, and
     raises before mutating anything -- its guard loop precedes every write).
 C8  timing and memory at 2 x 2 with this instance's block size: per-solve wall time
     (network solves, wrapped at `network.Network.run_smopf`), initialization vs per-cycle
     time, peak RSS of the process tree, and the built block size (Var/Constraint counts of
     one TSO and one DSO block AFTER the ADMM setup). Preliminary data point, measured WITH
     THE QUADRATIC ACTIVE and at REDUCED years (1) -- days are unreduced.

If any branch raises, or produces an empty/inverted/non-finite interval, the stage records
the traceback and the offending entries and exits non-zero. It NEVER patches production.

================================================================================
EXACT COMMAND (repo root; canonical interpreter; attached; alone; both streams captured)
================================================================================
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
      p515_s50_w37_multiscenario_smoke.py --label w37_2x2_r1 \
      > data/SRP1/Results/P515S50/multiscenario_smoke/w37_2x2_r1_launch.log 2>&1

OUTPUT (write-once): data/SRP1/Results/P515S50/multiscenario_smoke/<label>/
Exit 0 when every check passes, 1 otherwise, 2 refused, 97 watchdog abort.
"""

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

# The bounded profile, declared and INSTALLED BEFORE ANY PRODUCTION IMPORT. The literal is
# `p514_n_instrumented_cstar.PERMITTED`; it is stated here rather than imported so that the
# guard is in force for the imports themselves, and it is CHECKED against that module's own
# constant immediately after the import below.
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]
GUARD = SolveProfileGuard(PERMITTED, label='P5.15 W37 2x2 multiscenario smoke').install()

import pyomo.environ as pe  # noqa: E402
import psutil  # noqa: E402

import definitions as DEF  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import network as network_module  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s41_hull_polish as HP  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- PRODUCTION_FILES_TO_CHECK_CLEAN

if [tuple(p) for p in N.PERMITTED] != [tuple(p) for p in PERMITTED]:
    raise RuntimeError(f'declared PERMITTED {PERMITTED} != p514_n.PERMITTED {N.PERMITTED}')

STAGE = ('P5.15 Addendum 36 W37 -- 2x2 multi-scenario SMOKE of the >1x1 hull-polish and '
         'settlement branches (quadratic scenario-deviation penalty active; NOT the pilot)')
SCHEMA = 'p515_s50_w37_multiscenario_smoke_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 36 (W37 task text)',
             'commit cb545653 (W35 item 3: the four single-scenario code paths generalized)',
             'W36: the signed linear row 18 is NOT implemented; the quadratic '
             'scenario-deviation penalty is what exists -- the formulation decision is PENDING']

WARNING_LABEL = ('SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY '
                 '(definitions.PENALTY_SCENARIO_DEVIATION = 9e4 on voltage + interface power, '
                 'definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4 on the shared ESS, '
                 'added to model.objective.expr and therefore OUTSIDE Q(x)); '
                 'NOT THE PILOT INSTANCE; THE ROW-18 / alpha DECISION IS PENDING. '
                 'Branch-execution and sanity evidence only -- not a result, not comparable '
                 'with any SRP1 (1x1) figure.')

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S50', 'multiscenario_smoke')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s50_w37_smoke.lock')

# instance
OVERRIDE_YEARS = {'2025': 5}
OVERRIDE_MARKET_SCENARIOS = 2
OVERRIDE_OPERATION_SCENARIOS = 2
INSTANCE_LABEL = 'w37_2x2_smoke'
CYCLES = 2
REQUIRED_CONSECUTIVE_CYCLES = 10     # case-file value; the config hook checks the spec matches it
ACTIVE_NODES = (5, 7, 9)
UNIT1 = (0.25, 1.0)                  # p515_s50_marginal_campaign.UNIT1 -- the smallest node-7 4 h unit
UNIT_NODE = 7
ARMS = ('x0', 'unit')

# tolerances, stated here and carried into every artifact
TOL_DECOMPOSITION_REL = 1e-9
TOL_FAMILY_EXPECTATION_REL = 1e-9
TOL_S31C_RECONCILIATION_REL = 1e-9
TOL_PRICE_REL = 1e-12

GIB = 1 << 30
RSS_LIMIT_GIB = 12.0                 # this instance is ~1/5 of SRP1's block count at 4x the
                                     # scenario width; 12 GiB is a machine-protection ceiling,
                                     # not a measurement (recorded in launch.json)

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2

# The per-arm declared solve count, derived from the planning instance BEFORE the run and
# read back after it by the reconciliation (so the declaration cannot drift from the gate).
DECLARED = {}

THREAD_CAP_ENV = dict(S.THREAD_CAP_ENV)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(x):
    return isinstance(x, float) and math.isfinite(x)


# ==============================================================================
#  preconditions
# ==============================================================================
def _check_preconditions(out_dir):
    failures = []
    for path in (OWN_LOCK_PATH, G.CAMPAIGN_LOCK_PATH, os.path.join(REPO, '.p515_g_gate.lock'),
                 os.path.join(REPO, '.p515_s44_scale_measurement.lock')):
        if os.path.exists(path):
            failures.append(f'lock file exists: {path}')
    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')
    me = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me or 'python' not in cmd:
            continue
        if S.HARNESS_PATTERN.search(cmd):
            failures.append(f'another p51x/p514 harness is alive: pid={proc.info["pid"]} {cmd[:200]}')
    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(CP.PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')
    bad_env = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        failures.append(f'thread caps not in force (export them before launching): {bad_env}')
    return failures


# ==============================================================================
#  apply_common_values tripwire (C7)
# ==============================================================================
class _ApplyCommonValuesTripwire:
    """Counts EVERY call to `p56a_oracle.apply_common_values` for the whole run.

    The claim is "it is not on this path". CLAUDE.md's sixth rule says such a claim is
    ARMED, never asserted -- so the callable is wrapped, not inspected. `probe` is set only
    around the ONE deliberate probe call at the very end; any call outside that window is a
    violation and is recorded with its stack."""

    def __init__(self):
        self.calls_unexpected = []
        self.calls_probe = 0
        self.probe = False
        self._original = None

    def install(self):
        self._original = O.apply_common_values
        tripwire = self

        def wrapped(*args, **kwargs):
            if tripwire.probe:
                tripwire.calls_probe += 1
            else:
                tripwire.calls_unexpected.append(''.join(traceback.format_stack(limit=25)))
            return tripwire._original(*args, **kwargs)

        O.apply_common_values = wrapped
        return self

    def uninstall(self):
        if self._original is not None:
            O.apply_common_values = self._original


# ==============================================================================
#  per-solve timing (C8)
# ==============================================================================
class SolveTimer:
    """Wall time of every `network.Network.run_smopf` call, tagged by agent/year/day."""

    def __init__(self):
        self.records = []
        self._original = None

    def install(self):
        self._original = network_module.Network.run_smopf
        timer = self

        def wrapped(self_net, *args, **kwargs):
            t0 = time.time()
            try:
                return timer._original(self_net, *args, **kwargs)
            finally:
                timer.records.append({
                    'network': self_net.name, 'year': str(self_net.year), 'day': str(self_net.day),
                    'is_transmission': bool(getattr(self_net, 'is_transmission', False)),
                    'phase': STAGE_PHASE['name'], 'wall_s': time.time() - t0})
        network_module.Network.run_smopf = wrapped
        return self

    def uninstall(self):
        if self._original is not None:
            network_module.Network.run_smopf = self._original

    @staticmethod
    def _stats(times):
        times = sorted(times)
        if not times:
            return {'n': 0}
        return {'n': len(times), 'total_s': sum(times), 'mean_s': sum(times) / len(times),
                'min_s': times[0], 'median_s': times[len(times) // 2], 'max_s': times[-1]}

    def summary(self, phase_filter=None):
        rows = [r for r in self.records if phase_filter is None or r['phase'] == phase_filter]
        out = self._stats([r['wall_s'] for r in rows])
        out['by_agent'] = {
            'TSO': self._stats([r['wall_s'] for r in rows if r['is_transmission']]),
            'DSO': self._stats([r['wall_s'] for r in rows if not r['is_transmission']])}
        out['scope_note'] = ('NETWORK solves only -- wrapped at network.Network.run_smopf. The '
                             'ESSO solves (3 per round) go through '
                             'shared_energy_storage_data._run_solver_attempt and are NOT in this '
                             'table; they ARE in the guard counts.')
        return out


STAGE_PHASE = {'name': 'start'}


def _set_phase(name):
    STAGE_PHASE['name'] = name
    S.STAGE_REF['name'] = name


# ==============================================================================
#  block size (C8)
# ==============================================================================
def _block_size(model):
    n_var = sum(len(v) for v in model.component_objects(pe.Var, active=True, descend_into=True))
    n_con = sum(len(c) for c in model.component_objects(pe.Constraint, active=True, descend_into=True))
    n_expr = sum(len(e) for e in model.component_objects(pe.Expression, active=True, descend_into=True))
    return {'n_var_data': n_var, 'n_constraint_data': n_con, 'n_expression_data': n_expr,
            'n_market_scenarios': len(model.scenarios_market),
            'n_operation_scenarios': len(model.scenarios_operation),
            'n_periods': len(model.periods)}


# ==============================================================================
#  the checks
# ==============================================================================
def check_expectation_mode(planning, models):
    """C1: every block is >1 x 1 and every generalized site takes the expectation branch."""
    out = {'blocks': {}, 'all_multi_scenario': True}
    tso = planning.transmission_network
    for year in tso.years:
        for day in tso.days:
            m = models['tso'][year][day]
            single = O._block_is_single_scenario(m)
            out['blocks'][f'TSO|{year}|{day}'] = {
                'n_market': len(m.scenarios_market), 'n_operation': len(m.scenarios_operation),
                'single_scenario': single,
                'has_scenario_deviation_penalty': hasattr(m, 'scenario_deviation_penalty')}
            out['all_multi_scenario'] &= not single
    for node, dso in sorted(planning.distribution_networks.items()):
        for year in dso.years:
            for day in dso.days:
                m = models['dso'][node][year][day]
                single = O._block_is_single_scenario(m)
                out['blocks'][f'DSO|{node}|{year}|{day}'] = {
                    'n_market': len(m.scenarios_market), 'n_operation': len(m.scenarios_operation),
                    'single_scenario': single,
                    'has_scenario_deviation_penalty': hasattr(m, 'scenario_deviation_penalty')}
                out['all_multi_scenario'] &= not single
    return out


CHANNEL_AGENTS = {
    'V': ('tso_v', 'dso_v'),
    'PF_P': ('tso_p', 'dso_p'),
    'PF_Q': ('tso_q', 'dso_q'),
    'ESS_P': ('tso_sess_p', 'dso_sess_p', 'esso_sess_p'),
    'ESS_Q': ('tso_sess_q', 'dso_sess_q', 'esso_sess_q'),
}


def check_hull_intervals(hull, descriptors_detail):
    """C2: finite, non-empty, EXACTLY the closed hull of the agents' achieved values,
    including the ESSO endpoint on the two shared-ESS channels."""
    per_channel = {}
    violations = []
    for channel, fields in CHANNEL_AGENTS.items():
        n = 0
        n_degenerate = 0
        n_esso_binding = 0
        n_esso_interior = 0
        width_min = None
        width_max = None
        for key, entry in hull.items():
            values = [entry.get(f) for f in fields]
            if not all(_finite(v) for v in values):
                violations.append({'check': 'finite_agent_values', 'channel': channel,
                                   'key': str(key), 'values': values})
                continue
            lo, hi = min(values), max(values)
            if not (_finite(lo) and _finite(hi)):
                violations.append({'check': 'finite_interval', 'channel': channel,
                                   'key': str(key), 'lo': lo, 'hi': hi})
            if hi < lo:
                violations.append({'check': 'non_empty_interval', 'channel': channel,
                                   'key': str(key), 'lo': lo, 'hi': hi})
            n += 1
            if lo == hi:
                n_degenerate += 1
            width = hi - lo
            width_min = width if width_min is None else min(width_min, width)
            width_max = width if width_max is None else max(width_max, width)
            if channel in ('ESS_P', 'ESS_Q'):
                esso = entry[fields[2]]
                if not (lo <= esso <= hi):
                    violations.append({'check': 'esso_endpoint_inside_hull', 'channel': channel,
                                       'key': str(key), 'lo': lo, 'hi': hi, 'esso': esso})
                if esso == lo or esso == hi:
                    n_esso_binding += 1
                else:
                    n_esso_interior += 1
        per_channel[channel] = {
            'n_entries': n, 'n_degenerate': n_degenerate,
            'width_min': width_min, 'width_max': width_max,
            'agent_fields': list(fields), 'n_agents': len(fields),
            'n_esso_is_a_binding_endpoint': (n_esso_binding if channel in ('ESS_P', 'ESS_Q') else None),
            'n_esso_strictly_interior': (n_esso_interior if channel in ('ESS_P', 'ESS_Q') else None),
        }

    # bitwise agreement with what apply_hull_bounds actually installed
    mismatches = []
    checked = 0
    for d in descriptors_detail:
        key = (d['node'], d['year'], d['day'], d['period'])
        entry = hull.get(key)
        if entry is None:
            # the JSON round trip stringifies nothing here (in-process dict), so this is a defect
            mismatches.append({'reason': 'no hull entry for descriptor', 'descriptor': dict(
                (k, d[k]) for k in ('channel', 'node', 'year', 'day', 'period', 'side'))})
            continue
        fields = CHANNEL_AGENTS[d['channel']]
        values = [entry[f] for f in fields]
        lo, hi = min(values), max(values)
        checked += 1
        if (lo, hi, lo == hi) != (d['lo'], d['hi'], d['degenerate']):
            mismatches.append({'channel': d['channel'], 'key': str(key), 'side': d['side'],
                               'recomputed': [lo, hi, lo == hi],
                               'installed': [d['lo'], d['hi'], d['degenerate']]})
    return {'per_channel': per_channel, 'n_violations': len(violations),
            'violations': violations[:50], 'n_descriptors_checked': checked,
            'n_descriptor_mismatches': len(mismatches), 'descriptor_mismatches': mismatches[:50],
            'pass': not violations and not mismatches,
            'definition': ('per (node, year, day, period) and channel: lo = min(agent values), '
                           'hi = max(agent values); V/PF_P/PF_Q over 2 agents (TSO, DSO), '
                           'ESS_P/ESS_Q over 3 (TSO, DSO, ESSO -- Planner ruling). Recomputed '
                           'here and compared for BITWISE equality with the interval '
                           'p515_s41_hull_polish.apply_hull_bounds actually installed.')}


def check_decomposition(planning, models, per_block):
    """C4: sum(families) == weighted_base_objective - weight * interface_settlement, and each
    family equals its own probability-weighted per-scenario values."""
    rows = []
    worst_rel = 0.0
    worst_family_rel = 0.0
    n_bad = 0
    for key, block in per_block.items():
        # `p56a_oracle._tagged_holders` keys: 'TSO|<year>|<day>' and 'DSO<node>|<year>|<day>'
        tag, year, day = key.split('|')
        if tag == 'TSO':
            holder = planning.transmission_network
            model = models['tso'][int(year)][day]
            network = holder.network[int(year)][day]
        else:
            node = int(tag[3:])
            holder = planning.distribution_networks[node]
            model = models['dso'][node][int(year)][day]
            network = holder.network[int(year)][day]
        weight = block['weight']
        families = block['families']
        settlement_local = srp._get_local_interface_settlement(model)
        lhs = sum(v for v in families.values() if v is not None)
        rhs = block['weighted_base_objective'] - weight * settlement_local
        scale = max(abs(block['weighted_base_objective']), 1.0)
        rel = abs(lhs - rhs) / scale
        worst_rel = max(worst_rel, rel)
        # each family's expectation vs its own per-scenario values
        family_rel = {}
        for name, value in families.items():
            per_scen = (block.get('families_per_scenario_unweighted') or {}).get(name)
            if value is None or not per_scen:
                family_rel[name] = None
                continue
            recomputed = weight * sum(
                network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                * per_scen[f'{s_m}_{s_o}']
                for s_m in model.scenarios_market for s_o in model.scenarios_operation)
            fscale = max(abs(value), abs(recomputed), 1.0)
            family_rel[name] = abs(value - recomputed) / fscale
            worst_family_rel = max(worst_family_rel, family_rel[name])
        bad = rel > TOL_DECOMPOSITION_REL or any(
            v is not None and v > TOL_FAMILY_EXPECTATION_REL for v in family_rel.values())
        n_bad += int(bad)
        rows.append({'block': key, 'weight': weight,
                     'weighted_base_objective': block['weighted_base_objective'],
                     'sum_families': lhs,
                     'interface_settlement_local_unweighted': settlement_local,
                     'weighted_interface_settlement': weight * settlement_local,
                     'target_base_minus_settlement': rhs,
                     'abs_diff': lhs - rhs, 'relative': rel,
                     'family_expectation_relative': family_rel,
                     'n_scenario_keys': len(
                         (block.get('families_per_scenario_unweighted') or {}).get(
                             'generation_cost') or {}),
                     'within_tolerance': not bad})
    return {'identity': ('sum(families) == weighted_base_objective - weight * '
                         'interface_settlement_weight * interface_settlement (the seven family '
                         'aggregates of model_construction_helpers.objective_function_rule; the '
                         'settlement term is the eighth and is not among the families)'),
            'tolerance_relative': TOL_DECOMPOSITION_REL,
            'family_expectation_identity': ('families[name] == weight * sum_{s_m,s_o} '
                                            'prob_market[s_m] * prob_operation[s_o] * '
                                            'families_per_scenario_unweighted[name][s_m_s_o]'),
            'family_expectation_tolerance_relative': TOL_FAMILY_EXPECTATION_REL,
            'worst_relative': worst_rel, 'worst_family_relative': worst_family_rel,
            'n_blocks': len(rows), 'n_outside_tolerance': n_bad,
            'pass': n_bad == 0, 'per_block': rows}


def check_s31c(detail):
    """C5: the S31C priced-residual reconciliation, on the writer's own reading rule."""
    lhs = detail['t_tso_plus_t_dso_terminal']
    rhs = -1.0 * sum(v['sum_pi_baseMVA_residual_weighted']
                     for v in detail['interface_consensus_residual_per_dso'].values())
    scale = max(abs(lhs), abs(rhs), 1.0)
    rel = abs(lhs - rhs) / scale
    return {'identity': ('t_tso_plus_t_dso_terminal == -1 * sum over DSO nodes of '
                         'sum_pi_baseMVA_residual_weighted (the writer\'s own reading rule; '
                         'T_TSO = -prob*pi*baseMVA*p_TSO, T_DSO = +prob*pi*baseMVA*p_DSO)'),
            'tolerance_relative': TOL_S31C_RECONCILIATION_REL,
            't_tso_plus_t_dso_terminal': lhs,
            'minus_sum_priced_residual_weighted': rhs,
            'abs_diff': lhs - rhs, 'relative': rel,
            't_tso_total': detail['t_tso_total'],
            't_dso_by_node': detail['t_dso_by_node'],
            'pass': rel <= TOL_S31C_RECONCILIATION_REL}


def check_prices(planning, models, detail):
    """C6: price_per_mwh_by_scenario and the expectation-vs-per-scenario conventions."""
    reporting = detail['interface_reporting_detail']
    tso = planning.transmission_network
    n_entries = 0
    n_bad = 0
    worst_price_rel = 0.0
    spreads = []
    per_day = []
    problems = []
    for node_id, by_year in reporting.items():
        for year, by_day in by_year.items():
            for day, day_detail in by_day.items():
                y = int(year)
                network = tso.network[y][day]
                model = models['tso'][y][day]
                n_m = len(model.scenarios_market)
                n_o = len(model.scenarios_operation)
                naive = 0.0
                for p_key, period in day_detail['periods'].items():
                    p = int(p_key)
                    n_entries += 1
                    price = period['price_per_mwh']
                    by_scen = period['price_per_mwh_by_scenario']
                    expected = float(sum(network.prob_market_scenarios[s_m]
                                         * network.cost_energy_p[s_m][p]
                                         for s_m in model.scenarios_market))
                    rel = abs(price - expected) / max(abs(expected), 1.0)
                    worst_price_rel = max(worst_price_rel, rel)
                    market_values = [float(network.cost_energy_p[s_m][p])
                                     for s_m in model.scenarios_market]
                    spread = max(market_values) - min(market_values)
                    spreads.append(spread)
                    ok = (_finite(price) and len(by_scen) == n_m * n_o
                          and all(_finite(v) for v in by_scen.values())
                          and rel <= TOL_PRICE_REL
                          and min(market_values) - 1e-12 <= price <= max(market_values) + 1e-12
                          and all(by_scen[f'{s_m}_{s_o}'] == by_scen[f'{s_m}_0']
                                  for s_m in model.scenarios_market
                                  for s_o in model.scenarios_operation))
                    if not ok:
                        n_bad += 1
                        if len(problems) < 25:
                            problems.append({'node': node_id, 'year': year, 'day': day,
                                             'period': p, 'price': price,
                                             'expected_price': expected, 'relative': rel,
                                             'by_scenario': by_scen,
                                             'n_keys_expected': n_m * n_o})
                    naive += price * period['p_int_dso_expected_mw']
                per_day.append({
                    'node': node_id, 'year': year, 'day': day,
                    'dso_settlement_sum_pi_p_int_multi_market_branch':
                        day_detail['dso_settlement_sum_pi_p_int'],
                    'naive_sum_expected_price_times_expected_p_int': naive,
                    'difference_covariance_term':
                        day_detail['dso_settlement_sum_pi_p_int'] - naive,
                    'n_market_scenarios': day_detail['n_market_scenarios'],
                    'n_operation_scenarios': day_detail['n_operation_scenarios']})
    spreads_sorted = sorted(spreads)
    return {'n_entries': n_entries, 'n_bad': n_bad, 'problems': problems,
            'worst_price_relative_vs_recomputed_expectation': worst_price_rel,
            'price_tolerance_relative': TOL_PRICE_REL,
            'market_price_spread_over_scenarios': {
                'min': spreads_sorted[0] if spreads_sorted else None,
                'median': spreads_sorted[len(spreads_sorted) // 2] if spreads_sorted else None,
                'max': spreads_sorted[-1] if spreads_sorted else None,
                'n_zero': sum(1 for s in spreads if s == 0.0)},
            'settlement_branch': per_day,
            'settlement_branch_note': (
                'dso_settlement_sum_pi_p_int is taken INSIDE the expectation above one market '
                'scenario (sum_{s_m,s_o} prob * cost_energy_p[s_m][t] * pg_adn[s_m,s_o,t] * '
                'baseMVA); the naive column is sum_t E[pi_t] * E[p_int,t], which the branch '
                'deliberately is NOT. A nonzero difference is the price-quantity covariance and '
                'is EVIDENCE that the multi-market branch executed.'),
            'pass': n_bad == 0 and any(s > 0.0 for s in spreads)}


# ==============================================================================
#  the per-arm post-run hook
# ==============================================================================
def make_hook(arm, arm_dir, holder, timer):
    def hook(planning, sed, models, rows, report, out_dir, label, state=None):
        record = holder.setdefault(arm, {})
        record['label'] = label
        record['out_dir'] = os.path.relpath(out_dir, REPO)
        try:
            if state is None or 'consensus_vars' not in state:
                raise RuntimeError('state/consensus_vars not available to post_run_hook')

            _set_phase(f'{arm}: C1 expectation-mode')
            record['c1_expectation_mode'] = check_expectation_mode(planning, models)

            _set_phase(f'{arm}: block size')
            year0 = next(iter(planning.transmission_network.years))
            day0 = next(iter(planning.transmission_network.days))
            node0 = sorted(planning.distribution_networks)[0]
            record['block_size'] = {
                f'TSO|{year0}|{day0}': _block_size(models['tso'][year0][day0]),
                f'DSO{node0}|{year0}|{day0}': _block_size(models['dso'][node0][year0][day0])}

            _set_phase(f'{arm}: S31C settlement detail + writer')
            t0 = time.time()
            s31c_path = G.write_interface_settlement_detail_s31c(
                planning, sed, models, rows, report, out_dir, label)
            record['s31c_path'] = os.path.relpath(s31c_path, REPO)
            with open(s31c_path) as handle:
                s31c = json.load(handle)
            record['s31c_writer_wall_s'] = time.time() - t0
            record['c5_s31c_reconciliation'] = check_s31c(s31c)
            record['c6_prices'] = check_prices(planning, models, s31c)

            _set_phase(f'{arm}: common_coordinated_values')
            common = O.common_coordinated_values(planning, models, state['consensus_vars'])
            sample_key = sorted(common)[0]
            record['c1_common_coordinated_values'] = {
                'n_entries': len(common),
                'all_expectation_mode': all(e['expectation_mode'] for e in common.values()),
                'source_agent_values_distinct': sorted(
                    {e['source_agent_values'] for e in common.values()}),
                'source_interface_distinct': sorted({e['source_interface'] for e in common.values()}),
                'source_sess_distinct': sorted({e['source_sess'] for e in common.values()}),
                'sample_key': str(sample_key), 'sample_entry': common[sample_key],
                'n_nonfinite': sum(1 for e in common.values() for k, v in e.items()
                                   if isinstance(v, float) and not math.isfinite(v))}

            _set_phase(f'{arm}: hull_entries_with_esso')
            hull = HP.hull_entries_with_esso(planning, models, state['consensus_vars'])
            record['hull_n_entries'] = len(hull)

            _set_phase(f'{arm}: per_block_base_objectives (pre-polish)')
            per_block_before = O.per_block_base_objectives(planning, models)
            record['c4_decomposition'] = check_decomposition(planning, models, per_block_before)

            _set_phase(f'{arm}: hull polish (apply_hull_bounds + re-solve every block)')
            t0 = time.time()
            polish, hull_bound_detail = HP._polish_all_blocks_hull(
                planning, models, state['consensus_vars'])
            polish['runtime_s'] = time.time() - t0
            record['polish'] = polish
            record['c3_polish_solved'] = {
                'n_blocks': polish['n_blocks'], 'all_solved': polish['all_solved'],
                'failed_blocks': polish['failed_blocks'],
                'solve_profile': polish['solve_profile'],
                'gate_evaluated': polish['gate'] is not None,
                'gate': polish['gate'],
                'gate_caveat': ('the s41 gate threshold and FLAG_ABS are SRP1 C* constants '
                                '(1e-6 x 650,966,975.29); they are reported here for provenance '
                                'and are NOT a meaningful gate on this instance. Delta <= 0 also '
                                'need not hold above 1 x 1: the polish minimises model.objective, '
                                'whose .expr carries the quadratic scenario-deviation penalty, '
                                'while Delta is measured on objective_function_rule, which does '
                                'not.')}
            record['c2_hull_intervals'] = check_hull_intervals(hull, hull_bound_detail)

            detail_path = os.path.join(out_dir, 'hull_bound_detail.json')
            G._refuse_overwrite(detail_path)
            with open(detail_path, 'w') as handle:
                json.dump(hull_bound_detail, handle, indent=1, default=str)
            record['hull_bound_detail_path'] = os.path.relpath(detail_path, REPO)

            hull_path = os.path.join(out_dir, 'hull_entries.json')
            G._refuse_overwrite(hull_path)
            with open(hull_path, 'w') as handle:
                json.dump({str(k): v for k, v in hull.items()}, handle, indent=1, default=str)
            record['hull_entries_path'] = os.path.relpath(hull_path, REPO)

            # ---- C7, LAST: the deliberate probe. Raises before mutating anything. ----
            _set_phase(f'{arm}: C7 apply_common_values probe')
            TRIPWIRE.probe = True
            probe = {'called': True, 'raised': None, 'message': None}
            try:
                O.apply_common_values(planning, models, common)
                probe['raised'] = None
            except NotImplementedError as error:
                probe['raised'] = 'NotImplementedError'
                probe['message'] = str(error)
            except BaseException as error:  # noqa: BLE001 -- recorded, not hidden
                probe['raised'] = type(error).__name__
                probe['message'] = str(error)
            finally:
                TRIPWIRE.probe = False
            probe['pass'] = probe['raised'] == 'NotImplementedError'
            record['c7_apply_common_values_probe'] = probe
            record['hook_ok'] = True
        except BaseException as error:  # noqa: BLE001 -- recorded with its traceback, never hidden
            record['hook_ok'] = False
            record['hook_error'] = f'{type(error).__name__}: {error}'
            record['hook_traceback'] = traceback.format_exc()
            print(f'[W37 {arm}] POST-RUN HOOK RAISED:\n{record["hook_traceback"]}',
                  file=sys.stderr, flush=True)
        finally:
            _set_phase(f'{arm}: hook done')
    return hook


# ==============================================================================
#  one arm
# ==============================================================================
def run_arm(arm, label_root, out_dir, planning0, holder, timer, watchdog):
    arm_dir = os.path.join(out_dir, f'arm_{arm}')
    os.makedirs(arm_dir)
    if arm == 'x0':
        investment_map = {node: (0.0, 0.0) for node in ACTIVE_NODES}
    else:
        investment_map = {node: (UNIT1 if node == UNIT_NODE else (0.0, 0.0))
                          for node in ACTIVE_NODES}
    label = f'w37_{arm}'
    # the P56A working-dir id is NEVER reusable (network logs append), so it carries the RUN
    # label as well as the arm -- a second run under a new label cannot collide with a first.
    eval_id = f'p515s50_w37_{label_root}_{arm}'
    record = holder.setdefault(arm, {})
    record['investment_map'] = {str(k): list(v) for k, v in investment_map.items()}
    record['investment_year'] = N.INVEST_YEAR
    record['eval_id'] = eval_id

    spec_like = {'configuration': {'overrides': {},
                                   'case_file_anderson_acceleration': dict(S.CASE_FILE_AA)},
                 'cap': CYCLES,
                 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}
    cfg_holder = {}
    hook_in = H._config_hook_factory(spec_like, cfg_holder, overrides={})
    pre_hook = S.snapshot_hook_wrapper(hook_in, 'off', record)

    _set_phase(f'{arm}: run_admm_arm (initialization + {CYCLES} cycles)')
    t0 = time.time()
    n_solves_before = GUARD.counts['permitted_solve']
    report, report_path = G.run_admm_arm(
        label, arm_dir, k_override=None, investment_map=investment_map,
        num_max_iters_override=CYCLES, eval_id=eval_id, apply_rho=False,
        full_diagnostics_in_rows=True, pre_solve_hook=pre_hook,
        post_run_hook=make_hook(arm, arm_dir, holder, timer))
    record['arm_wall_s'] = time.time() - t0
    record['solves_in_arm_including_polish'] = GUARD.counts['permitted_solve'] - n_solves_before
    record['g_report_path'] = os.path.relpath(report_path, REPO)
    record['anderson_acceleration_effective'] = cfg_holder.get('anderson_acceleration_effective')
    record['configuration_checks'] = cfg_holder.get('configuration_checks')
    record['cycles_run'] = report['cycles_run']
    record['converged_at_cycle'] = report.get('converged_at_cycle')
    record['recourse'] = report.get('recourse')
    record['gross_operational_cost'] = report.get('gross_operational_cost')
    record['objective_convention'] = (
        "recourse = net_operational_recourse (settlement-excluded, salvage-netted); "
        "gross_operational_cost = settlement-excluded gross; both as "
        "shared_resources_planning._get_operational_recourse_components defines them")
    record['cycle_trajectory'] = report.get('cycle_trajectory')
    record['network_failures_summary'] = report.get('network_failures_summary')
    record['arm_solve_profile'] = report.get('solve_profile')
    record['event_level_reconciliation'] = S.event_level_solve_reconciliation(
        report, S.declared_solve_profile(planning0, CYCLES)['declared_base_solves'])
    record['watchdog_peak_after_arm'] = dict(watchdog.peak)
    record['solve_times'] = {
        'all': timer.summary(),
        'by_phase': {phase: timer.summary(phase) for phase in sorted(
            {r['phase'] for r in timer.records})}}
    return record


# ==============================================================================
#  main
# ==============================================================================
TRIPWIRE = _ApplyCommonValuesTripwire()


def main():
    global CYCLES
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--cycles', type=int, default=CYCLES)
    parser.add_argument('--arms', default=','.join(ARMS),
                        help='comma-separated subset of ' + ','.join(ARMS))
    parser.add_argument('--preflight-only', action='store_true',
                        help=('derive the case, read the planning problem, install the oracle '
                              'baseline and run the rule-eleven checklist, then STOP. Zero '
                              'solves (the guard is verified at exactly 0).'))
    args = parser.parse_args()
    CYCLES = args.cycles
    arms = tuple(a.strip() for a in args.arms.split(',') if a.strip())
    for a in arms:
        if a not in ARMS:
            parser.error(f'unknown arm {a!r}')

    out_dir = os.path.join(OUT_ROOT, args.label)
    failures = _check_preconditions(out_dir)
    if failures:
        for f in failures:
            print(f'[W37 PRECONDITION FAILED] {f}', file=sys.stderr)
        return EXIT_REFUSED
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: lock held {OWN_LOCK_PATH}', file=sys.stderr)
        return EXIT_REFUSED
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'label': args.label, 'started_utc': _utc()}, handle)

    started = time.time()
    holder = {}
    summary = {}
    timer = SolveTimer().install()
    TRIPWIRE.install()
    watchdog = None
    try:
        os.makedirs(out_dir)
        watchdog = S.Watchdog(out_dir, 'w37', limit_bytes=int(RSS_LIMIT_GIB * GIB))
        watchdog.start()
        stages = S.StageLog(os.path.join(out_dir, 'stages_w37.jsonl'), watchdog)

        # ---- derived case ----
        _set_phase('derive case')
        case_dir = os.path.join(out_dir, 'case')
        os.makedirs(case_dir)
        case, spec, changes = S.derive_case('srp1', {
            'years': OVERRIDE_YEARS,
            'num_market_scenarios': OVERRIDE_MARKET_SCENARIOS,
            'num_operation_scenarios': OVERRIDE_OPERATION_SCENARIOS})
        case_path = os.path.join(case_dir, 'SRP1__w37_2x2.json')
        with open(case_path, 'w') as handle:
            json.dump(case, handle, indent='\t')
        derived = {'path': os.path.relpath(case_path, REPO), 'sha256': _sha256_file(case_path),
                   'source': os.path.relpath(S.SOURCE_CASE, REPO),
                   'source_sha256': _sha256_file(S.SOURCE_CASE),
                   'changes_vs_source': changes,
                   'days_note': 'Days left UNCHANGED (4 days); derive_case does not edit them'}

        launch = {
            'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
            'WARNING': WARNING_LABEL,
            'label': args.label, 'instance': INSTANCE_LABEL, 'instance_definition': spec,
            'derived_case': derived, 'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': S._git(['rev-parse', 'HEAD']),
            'git_tracked_changes': S._git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
            'cycles': CYCLES, 'arms': list(arms),
            'guard_permitted': [list(p) for p in PERMITTED],
            'case_file_anderson_acceleration_declared': dict(S.CASE_FILE_AA),
            'ess_params_file': {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
                                'sha256': _sha256_file(os.path.join(
                                    REPO, 'data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')),
                                'note': 'the committed ageing BASELINE (W21, Addendum 30)'},
            'snapshots': 'off',
            'watchdog': {'rss_limit_bytes': int(RSS_LIMIT_GIB * GIB), 'rss_limit_gib': RSS_LIMIT_GIB,
                         'gating_measure': S.GATING_MEASURE_DEFAULT,
                         'note': 'machine protection only; not a measurement'},
            'quadratic_scenario_deviation_penalty': {
                'PENALTY_SCENARIO_DEVIATION': DEF.PENALTY_SCENARIO_DEVIATION,
                'PENALTY_SHARED_ESS_SCENARIO_DEVIATION':
                    DEF.PENALTY_SHARED_ESS_SCENARIO_DEVIATION,
                'added_to': 'model.objective.expr (NOT objective_function_rule / Q(x))'},
            'machine': {'total_memory_bytes': psutil.virtual_memory().total,
                        'cpu_count': os.cpu_count()},
            'started_utc': _utc(), 'pid': os.getpid(),
        }

        # ---- read the planning problem from the derived case ----
        planning0 = S.read_planning_from_derived_case(launch, out_dir, stages)
        checksum = S.inject_oracle_baseline(O, planning0, launch)
        launch['scenario_checksum'] = checksum
        launch['scenario_checksum_note'] = (
            'a NEW realization (different years and scenario counts -> different derived seeds); '
            'never compared with the SRP1 canonical checksum -- install_baseline was called with '
            f'instance_label={INSTANCE_LABEL!r}')
        launch['planning_dimensions'] = S.planning_dimensions(planning0)
        launch['expected_block_counts'] = S.expected_block_counts(planning0)
        # canonical solver / environment identity (the repository provenance gate). The scenario
        # checksum is EXPECTED to differ on a derived instance and is not gated on; any OTHER
        # identity failure (wrong IPOPT, wrong interpreter, ...) stops the run.
        prov = S.provenance_record(planning0, INSTANCE_LABEL, checksum)
        launch['provenance'] = prov
        non_checksum = [f for f in prov['gate_failures'] if f['identity'] != 'scenario checksum']
        if non_checksum:
            raise RuntimeError(f'provenance: non-canonical identity: {non_checksum}')

        declared = S.declared_solve_profile(planning0, CYCLES)
        n_network_blocks = launch['expected_block_counts']['network_blocks']
        declared_total = len(arms) * (declared['declared_base_solves'] + n_network_blocks)
        DECLARED['per_arm'] = declared['declared_base_solves'] + n_network_blocks
        launch['declared_solve_profile'] = {
            **declared,
            'polish_solves_per_arm': n_network_blocks,
            'n_arms': len(arms),
            'declared_total_strict': declared_total,
            'declared_per_arm': DECLARED.get('per_arm'),
            'declared_per_arm_note': ('= [(1 + n_dso) * n_years * n_days + n_esso] * (cycles + 1) '
                                      '+ (1 + n_dso) * n_years * n_days polish solves; recorded '
                                      'in launch.json before the run'),
            'gated_identity': ('observed == declared_total_strict + sum over arms of '
                               '[ADMM retries credited per failure event] + [polish inner-guard '
                               'surplus over one solve per block]'),
        }

        # ---- rule eleven: capture paths, BEFORE the run ----
        _set_phase('rule eleven checklist')
        checklist = {
            'O.common_coordinated_values': callable(getattr(O, 'common_coordinated_values', None)),
            'O.per_block_base_objectives': callable(getattr(O, 'per_block_base_objectives', None)),
            'O._block_is_single_scenario': callable(getattr(O, '_block_is_single_scenario', None)),
            'O._scenario_expectation': callable(getattr(O, '_scenario_expectation', None)),
            'O.apply_common_values (tripwired, must not be called)': TRIPWIRE._original is not None,
            'HP.hull_entries_with_esso': callable(getattr(HP, 'hull_entries_with_esso', None)),
            'HP.apply_hull_bounds': callable(getattr(HP, 'apply_hull_bounds', None)),
            'HP._polish_all_blocks_hull': callable(getattr(HP, '_polish_all_blocks_hull', None)),
            'srp._get_interface_reporting_detail': callable(
                getattr(srp, '_get_interface_reporting_detail', None)),
            'srp._expected_market_price': callable(getattr(srp, '_expected_market_price', None)),
            'srp._get_local_interface_settlement': callable(
                getattr(srp, '_get_local_interface_settlement', None)),
            'G.write_interface_settlement_detail_s31c': callable(
                getattr(G, 'write_interface_settlement_detail_s31c', None)),
            'mch.objective_function_rule': callable(getattr(mch, 'objective_function_rule', None)),
            'instance_is_multi_scenario': (planning0.num_market_scenarios > 1
                                           and planning0.transmission_network.num_oper_scenarios > 1),
            'every_network_is_multi_scenario': all(
                h.num_oper_scenarios > 1 for h in [planning0.transmission_network]
                + [planning0.distribution_networks[n] for n in planning0.distribution_networks]),
            'timer_installed': timer._original is not None,
            'watchdog_running': watchdog.is_alive(),
        }
        with stages.stage('assert_s31c_capture_paths (production rule eleven, zero solves)'):
            checklist['G.assert_s31c_capture_paths'] = G.assert_s31c_capture_paths(planning0)
        missing = [k for k, v in checklist.items() if v is False]
        if missing:
            raise RuntimeError(f'RULE ELEVEN (W37): capture paths missing -> {missing}')
        launch['rule_eleven_checklist'] = checklist
        S._write_once_json(os.path.join(out_dir, 'launch.json'), launch)
        print(f'[W37] launch written; instance {INSTANCE_LABEL} checksum {checksum[:16]} '
              f"blocks={launch['expected_block_counts']} declared_total={declared_total}", flush=True)

        # ---- arms ----
        if args.preflight_only:
            print('[W37] --preflight-only: stopping before any arm (declared 0 solves)', flush=True)
            arms = ()
        for arm in arms:
            print(f'[W37] arm {arm} starting', flush=True)
            with stages.stage(f'arm {arm}'):
                run_arm(arm, args.label, out_dir, planning0, holder, timer, watchdog)
            print(f"[W37] arm {arm} done: wall={holder[arm]['arm_wall_s']:.1f}s "
                  f"solves={holder[arm]['solves_in_arm_including_polish']} "
                  f"hook_ok={holder[arm].get('hook_ok')}", flush=True)
    except BaseException as error:  # noqa: BLE001
        summary['fatal_error'] = f'{type(error).__name__}: {error}'
        summary['fatal_traceback'] = traceback.format_exc()
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        GUARD.uninstall()
        timer.uninstall()
        TRIPWIRE.uninstall()
        if watchdog is not None:
            memory_final = watchdog.stop()
        else:
            memory_final = None
        try:
            os.remove(OWN_LOCK_PATH)
        except FileNotFoundError:
            pass

    # ---- reconcile the solve profile ----
    observed = GUARD.counts['permitted_solve']
    admm_retries = 0
    polish_surplus = 0
    unsupported = []
    for arm in holder:
        rec = holder[arm].get('event_level_reconciliation') or {}
        if rec.get('supported'):
            admm_retries += rec.get('retry_solves_credited', 0)
        else:
            unsupported.append({'arm': arm, 'reconciliation': rec})
        profile = ((holder[arm].get('polish') or {}).get('solve_profile') or {})
        polish_surplus += profile.get('retries_beyond_one_per_block', 0) or 0
    launch_path = os.path.join(out_dir, 'launch.json')
    declared_block = {}
    if os.path.exists(launch_path):
        with open(launch_path) as handle:
            declared_block = json.load(handle).get('declared_solve_profile', {})
    # the count GATED ON is derived from the arms that actually ran, at the per-arm
    # declaration made before the run (`DECLARED['per_arm']`, recorded in launch.json)
    declared_total = len(holder) * DECLARED.get('per_arm', 0)
    expected = (None if unsupported else declared_total + admm_retries + polish_surplus)
    guard_failures = (['event-level reconciliation UNSUPPORTED: ' + json.dumps(unsupported, default=str)]
                      if expected is None else GUARD.verify(expected))

    checks = {}
    for arm, rec in holder.items():
        checks[arm] = {
            'hook_ok': rec.get('hook_ok'),
            'c1_all_multi_scenario': (rec.get('c1_expectation_mode') or {}).get('all_multi_scenario'),
            'c1_all_expectation_mode': (rec.get('c1_common_coordinated_values') or {}).get(
                'all_expectation_mode'),
            'c2_hull_pass': (rec.get('c2_hull_intervals') or {}).get('pass'),
            'c3_polish_all_solved': (rec.get('c3_polish_solved') or {}).get('all_solved'),
            'c4_decomposition_pass': (rec.get('c4_decomposition') or {}).get('pass'),
            'c5_s31c_pass': (rec.get('c5_s31c_reconciliation') or {}).get('pass'),
            'c6_prices_pass': (rec.get('c6_prices') or {}).get('pass'),
            'c7_probe_pass': (rec.get('c7_apply_common_values_probe') or {}).get('pass'),
        }
    all_pass = (not summary.get('fatal_error') and not guard_failures
                and not TRIPWIRE.calls_unexpected
                and (bool(holder) or args.preflight_only)
                and all(all(v is True for v in c.values()) for c in checks.values()))

    summary.update({
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'WARNING': WARNING_LABEL,
        'label': args.label, 'instance': INSTANCE_LABEL, 'preflight_only': args.preflight_only,
        'launch': 'launch.json', 'cycles': CYCLES, 'arms': list(holder),
        'solve_profile': {
            'permitted': [list(p) for p in PERMITTED],
            'counts': dict(GUARD.counts),
            'declared_total_strict': declared_total,
            'declared_per_arm': DECLARED.get('per_arm'),
            'declared_per_arm_note': ('= [(1 + n_dso) * n_years * n_days + n_esso] * (cycles + 1) '
                                      '+ (1 + n_dso) * n_years * n_days polish solves; recorded '
                                      'in launch.json before the run'),
            'admm_retries_credited_per_event': admm_retries,
            'polish_retries_beyond_one_per_block': polish_surplus,
            'expected_gated': expected, 'observed': observed,
            'identity': (declared_block.get('gated_identity')
                         or 'preflight only: declared exactly 0 solves'),
            'verify_failures': guard_failures,
            'unsupported_reconciliations': unsupported,
            'permitted_sites': dict(GUARD.permitted_sites)},
        'apply_common_values_tripwire': {
            'n_unexpected_calls': len(TRIPWIRE.calls_unexpected),
            'unexpected_call_stacks': TRIPWIRE.calls_unexpected[:3],
            'n_probe_calls': TRIPWIRE.calls_probe,
            'declared': 'exactly 0 unexpected calls; exactly 1 probe call per arm, each raising '
                        'NotImplementedError'},
        'checks': checks, 'all_pass': all_pass,
        'memory': {'watchdog_peak': dict(watchdog.peak) if watchdog else None,
                   'memory_final': memory_final,
                   'ru_maxrss_self_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'ru_maxrss_children_ipopt_bytes':
                       resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   'units_note': 'ru_maxrss is in BYTES on macOS'},
        'solve_times_all_phases': timer.summary(),
        'solve_times_by_phase': {phase: timer.summary(phase)
                                 for phase in sorted({r['phase'] for r in timer.records})},
        'wall_clock_s': time.time() - started, 'ended_utc': _utc(),
    })
    summary['arms_detail'] = holder

    os.makedirs(out_dir, exist_ok=True)
    S._write_once_json(os.path.join(out_dir, 'w37_multiscenario_smoke.json'), summary)
    with open(os.path.join(out_dir, 'solve_times.json'), 'w') as handle:
        json.dump({'records': timer.records}, handle, indent=1, default=str)
    _write_markdown(out_dir, args.label, summary)
    _write_manifest(out_dir, args.label, holder)

    print(f'[W37] all_pass={all_pass} solves observed={observed} expected={expected} '
          f'verify_failures={guard_failures}', flush=True)
    return EXIT_OK if all_pass else EXIT_ERROR


def _write_markdown(out_dir, label, summary):
    lines = []
    a = lines.append
    a(f'# P5.15 Addendum 36 W37 -- 2x2 multi-scenario smoke ({label})')
    a('')
    a(f'> **{WARNING_LABEL}**')
    a('')
    a(f'- stage: {STAGE}')
    a(f'- instance label: `{summary["instance"]}`  (see `launch.json` for the derived case file, '
      'its sha256, and the scenario checksum)')
    a(f'- cycles per arm: {summary["cycles"]}  (certification NOT attempted)')
    a(f'- arms: {", ".join(summary["arms"])}')
    a(f'- wall clock: {summary["wall_clock_s"]:.1f} s')
    a('')
    a('## Solve profile (armed bounded guard, verified exactly)')
    sp = summary['solve_profile']
    a('')
    a(f'- declared strict total: **{sp["declared_total_strict"]}**')
    a(f'- ADMM retries credited per failure event: {sp["admm_retries_credited_per_event"]}')
    a(f'- polish retries beyond one per block: {sp["polish_retries_beyond_one_per_block"]}')
    a(f'- expected (gated): **{sp["expected_gated"]}**, observed: **{sp["observed"]}**')
    a(f'- verify failures: `{sp["verify_failures"]}`')
    a(f'- identity: {sp["identity"]}')
    a('')
    a('## Checks')
    a('')
    a('| arm | C1 multi-scenario | C1 expectation mode | C2 hull | C3 polish solved | '
      'C4 decomposition | C5 S31C | C6 prices | C7 probe |')
    a('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for arm, c in summary['checks'].items():
        a(f'| {arm} | {c["c1_all_multi_scenario"]} | {c["c1_all_expectation_mode"]} | '
          f'{c["c2_hull_pass"]} | {c["c3_polish_all_solved"]} | {c["c4_decomposition_pass"]} | '
          f'{c["c5_s31c_pass"]} | {c["c6_prices_pass"]} | {c["c7_probe_pass"]} |')
    a('')
    for arm, rec in summary['arms_detail'].items():
        a(f'## Arm `{arm}`')
        a('')
        a(f'- investment map (MVA, MWh) at year {rec.get("investment_year")}: '
          f'`{rec.get("investment_map")}`')
        a(f'- cycles run: {rec.get("cycles_run")}, converged at cycle: '
          f'{rec.get("converged_at_cycle")} (certification not attempted)')
        a(f'- objective convention: {rec.get("objective_convention")}')
        a(f'- recourse (net_operational_recourse): {rec.get("recourse")}')
        a(f'- gross_operational_cost (settlement-excluded): {rec.get("gross_operational_cost")}')
        a(f'- arm wall clock: {rec.get("arm_wall_s")} s; solves in arm incl. polish: '
          f'{rec.get("solves_in_arm_including_polish")}')
        bs = rec.get('block_size') or {}
        for key, val in bs.items():
            a(f'- block size `{key}`: {val}')
        c2 = rec.get('c2_hull_intervals') or {}
        a(f'- C2 hull: pass={c2.get("pass")}, descriptors checked={c2.get("n_descriptors_checked")}, '
          f'violations={c2.get("n_violations")}, descriptor mismatches='
          f'{c2.get("n_descriptor_mismatches")}')
        for channel, info in (c2.get('per_channel') or {}).items():
            a(f'  - {channel}: n={info["n_entries"]}, agents={info["n_agents"]}, '
              f'degenerate={info["n_degenerate"]}, width in [{info["width_min"]}, '
              f'{info["width_max"]}], ESSO binding={info["n_esso_is_a_binding_endpoint"]}, '
              f'ESSO interior={info["n_esso_strictly_interior"]}')
        c4 = rec.get('c4_decomposition') or {}
        a(f'- C4 decomposition: pass={c4.get("pass")}, worst relative={c4.get("worst_relative")}, '
          f'worst family-expectation relative={c4.get("worst_family_relative")}, '
          f'tolerance={c4.get("tolerance_relative")}')
        c5 = rec.get('c5_s31c_reconciliation') or {}
        a(f'- C5 S31C: pass={c5.get("pass")}, lhs={c5.get("t_tso_plus_t_dso_terminal")}, '
          f'rhs={c5.get("minus_sum_priced_residual_weighted")}, '
          f'relative={c5.get("relative")}, tolerance={c5.get("tolerance_relative")}')
        c6 = rec.get('c6_prices') or {}
        a(f'- C6 prices: pass={c6.get("pass")}, entries={c6.get("n_entries")}, bad={c6.get("n_bad")}, '
          f'worst relative={c6.get("worst_price_relative_vs_recomputed_expectation")}, '
          f'market spread={c6.get("market_price_spread_over_scenarios")}')
        c3 = rec.get('c3_polish_solved') or {}
        a(f'- C3 polish: blocks={c3.get("n_blocks")}, all_solved={c3.get("all_solved")}, '
          f'failed={c3.get("failed_blocks")}, solve profile={c3.get("solve_profile")}')
        a(f'  - gate caveat: {c3.get("gate_caveat")}')
        c7 = rec.get('c7_apply_common_values_probe') or {}
        a(f'- C7 apply_common_values probe: raised={c7.get("raised")}, pass={c7.get("pass")}')
        st = rec.get('solve_times') or {}
        a(f'- solve times (cumulative, all phases so far): {st.get("all")}')
        if rec.get('hook_error'):
            a('')
            a('```')
            a(rec.get('hook_traceback', ''))
            a('```')
        a('')
    a('## Timing and memory (C8) -- preliminary, quadratic active, 1 year x 4 days')
    a('')
    a(f'- per-solve wall time, all phases: `{summary["solve_times_all_phases"]}`')
    for phase, info in summary['solve_times_by_phase'].items():
        a(f'  - `{phase}`: {info}')
    a(f'- peak memory: `{summary["memory"]}`')
    a('')
    a('## What this does and does not establish')
    a('')
    a('- It establishes that the W35 item 3 >1 x 1 branches EXECUTE and produce internally '
      'consistent quantities on a real 2 x 2 instance, under the tolerances stated above.')
    a('- It does NOT establish any economic quantity, any convergence property, or anything '
      'about the pilot: the formulation it runs is the CURRENT quadratic scenario-deviation '
      'penalty, the row-18 / alpha decision is pending, and every number is at 1 representative '
      'year with 2 cycles and no certification.')
    path = os.path.join(out_dir, 'W37_MULTISCENARIO_SMOKE.md')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


def _write_manifest(out_dir, label, holder):
    files = {}
    for root, _dirs, names in os.walk(out_dir):
        for name in sorted(names):
            if name == 'manifest_sha256.json':
                continue
            path = os.path.join(root, name)
            files[os.path.relpath(path, REPO)] = {'sha256': _sha256_file(path),
                                                  'bytes': os.path.getsize(path)}
    work = {}
    for arm, rec in holder.items():
        eval_dir = os.path.join(O.WORK_DIR, rec.get('eval_id', ''))
        for root, _dirs, names in os.walk(eval_dir):
            for name in sorted(names):
                path = os.path.join(root, name)
                work[os.path.relpath(path, REPO)] = {'sha256': _sha256_file(path),
                                                     'bytes': os.path.getsize(path)}
    manifest = {'stage': STAGE, 'label': label, 'generated_utc': _utc(),
                'WARNING': WARNING_LABEL, 'files': files, 'p56a_working_dir_files': work}
    S._write_once_json(os.path.join(out_dir, 'manifest_sha256.json'), manifest)
    return manifest


if __name__ == '__main__':
    sys.exit(main())
