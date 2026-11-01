"""
P5.15 Step 3.7 (PLANNER_BRIEF_2026-09-13.md Addendum 22/23 item "Step 3.7 --
Anderson acceleration"): type-II Anderson acceleration (AA) on the joint
consensus/dual ADMM iterate.

Authority: frozen spec `data/SRP1/Results/P515S41/
frozen_s41_hull_aa_spec_v12_6e5a546f.json`, `item4_step_3_7_anderson`.
Citations: Walker & Ni (2011), "Anderson Acceleration for Fixed-Point
Iterations", SIAM J. Numer. Anal. 49(4); Zhang, O'Donoghue & Boyd (2020),
"Globally Convergent Type-I Anderson Acceleration for Nonsmooth Fixed-Point
Iterations", SIAM J. Optim. 30(4) (regularized least-squares form); Fu,
Zhang & Boyd (2020), "Anderson Accelerated Douglas-Rachford Splitting",
SIAM J. Sci. Comput. 42(6) (safeguard framework for ADMM/DR without extra
function evaluations).

DEFAULT OFF. `admm_parameters.anderson_acceleration['enabled']` (default
False, set in `ADMMParameters.__init__`) is the ONLY switch, checked via
`anderson_acceleration_enabled()`. With the flag off, `_run_operational_planning`
never calls anything in this module (see that function's `if aa_enabled:`
guards) -- the existing serial ADMM cycle is byte-for-byte unchanged. Not
wired to any case-file key; set programmatically only, exactly like
`persistent_workers`/`tso_snapshot_capture_mode` (`admm_parameters.py`,
`admm_persistent_workers.py`).

--------------------------------------------------------------------------
DESIGN (see WORKER_REPORT_S41_AA.md for the full write-up; this is the
short version for a reader of the code only).

Fixed point map. One full ADMM cycle is DSO solve -> consensus/dual update
-> TSO solve -> update -> ESSO solve -> update
(`shared_resources_planning.py:_run_operational_planning`, the `for iter in
range(...)` loop). Writing w = (z, u) for the stacked consensus/dual state
across the three channels (V, PF, ESS), one cycle computes w^{k+1} = F(w^k)
where F is exactly that DSO->TSO->ESSO sequence plus the existing
consensus/dual updates. g(w) := F(w) - w.

Iterate w = (z, u), per channel, in the SAME normalization as
`get_admm_boyd_residual_metrics` (`shared_resources_planning.py:6038`):

  V   (per node, year, day, period): z = TSO's interface voltage copy
      (`consensus_vars['vmag']['tso']['current']`), scaled by v_base
      (get_admm_boyd_residual_metrics:6163-6164). u = the DSO's stored
      dual (`dual_vars['vmag']['dso']['current']`) scaled by v_base
      (get_admm_boyd_residual_metrics:6161,6168 -- the `y_v` there),
      further divided by the channel's rho_v (the Boyd-convention SCALED
      dual u = y/rho; the codebase's own dual-ascent update,
      `_update_interface_power_flow_variables:7257-7259,7264-7269`, stores
      the UNSCALED multiplier y). The TSO's own stored dual copy
      (`dual_vars['vmag']['tso']['current']`) is not an independent
      component of w -- it equals -1 times the DSO's copy exactly, by
      construction of the plain update (see `write_back_w` docstring) --
      but IS written back so the TSO's own next-cycle read of it stays
      consistent.

  PF  (per node, year, day, period, power_type in {p, q}): z = TSO's
      interface flow copy (`consensus_vars['pf']['tso']['current']`),
      scaled by `interface_rating` (get_admm_boyd_residual_metrics:6192-6193).
      u = the DSO's stored dual (`dual_vars['pf']['dso']['current']`),
      scaled by the DSO's `s_base` (get_admm_boyd_residual_metrics:6197),
      divided by rho_pf. Same TSO-mirror rule as V.

  ESS (per node, year, day, period, power_type): ONE shared z
      (`consensus_vars['ess']['z']['current']`, the three-agent consensus
      variable), scaled by `2*S_ref` (S_ref = `admm_parameters.
      shared_ess_reference_rating_mva`, the Step-3.3(b)/3.4(d) FIXED
      reference rating -- this module REQUIRES it to be set; see
      `build_iterate_layout`). THREE per-agent scaled duals u_tso, u_dso,
      u_esso, one per block (`dual_vars['ess'][agent]['current']`), each
      divided by rho_ess (raw, no v_base/s_base factor -- matching the
      `y_agent` used directly, unnormalized, in
      get_admm_boyd_residual_metrics:6218,6228-6229). The per-agent x_i
      copies (`consensus_vars['ess']['tso'/'dso'/'esso']['current']`) are
      NOT part of w: in multi-block consensus ADMM they are the
      block-local primal minimizers recomputed fresh every cycle from
      (z^k, u_i^k), not carried state.

Scaled-dual (u = y/rho) precondition: rho is shared by ALL agents within a
channel (TSO, every DSO, and for ESS the ESSO too) -- `_update_admm_penalties`
(`shared_resources_planning.py:7099-7122`) applies the SAME multiplicative
factor to every agent's rho_v/rho_pf/rho_ess every cycle, and the case file
(`SRP1_params.json`, `admm.rho`) seeds them equal, so the invariant holds
for the life of a run started from the case file. The caller
(`shared_resources_planning._get_admm_rho_channel_scalars_for_aa`) asserts
this every cycle and raises rather than silently mis-scaling.

g = F(w) - w -- computed by the CALLER (`shared_resources_planning.py`) as
w_after - w_before, where w_before is collected at the TOP of the cycle
(before the DSO solve) and w_after right after the ESSO consensus/dual
update, using the SAME rho (rho only changes once, at the end of the cycle,
via `_update_admm_penalties`, so it is constant across one cycle's DSO/TSO/
ESSO solves).

Type-II AA (Walker & Ni 2011 Sec. 2.3; the "Type-II" name and the
regularized normal-equations solve follow Fu, Zhang & Boyd 2020 Sec. 2 and
Zhang, O'Donoghue & Boyd 2020 Sec. 2): memory m=5. With history
w_{k-m_k}, ..., w_k and g_{k-m_k}, ..., g_k (m_k = min(m, cycles seen - 1)),
build

  DeltaW = [w_{k-m_k+1}-w_{k-m_k}, ..., w_k - w_{k-1}]   (m_k columns)
  DeltaG = [g_{k-m_k+1}-g_{k-m_k}, ..., g_k - g_{k-1}]   (m_k columns)

  gamma* = argmin_gamma || g_k - DeltaG @ gamma ||^2 + lambda ||gamma||^2
         = solve( (DeltaG^T DeltaG + lambda I) gamma = DeltaG^T g_k )   (lambda = 1e-10, Tikhonov)

  w_hat = w_k + g_k - (DeltaW + DeltaG) @ gamma*

With m_k = 0 (first cycle, or right after any memory clear) DeltaW/DeltaG
have zero columns and w_hat reduces identically to the plain iterate
w_k + g_k -- this is how "AA active from cycle 1" is satisfied without a
special case (a cycle with no memory is numerically a no-op).

Safeguard (frozen spec `item4_step_3_7_anderson.method.safeguard`; the
literal ratchet form, NOT the Fu-Zhang-Boyd envelope -- the spec explicitly
permits either and asks for the choice to be recorded): let
`last_accepted_residual` be the combined scaled residual recorded
immediately before the LAST accepted AA step (initialized to +inf, so the
very first candidate, once memory allows one, is always tried). At cycle k,
`combined_residual_k = sqrt(sum over v,pf,ess of r_channel^2 + s_channel^2)`
(the Boyd primal+dual residual norms already computed by
`get_admm_boyd_residual_metrics` for every cycle regardless of AA -- no
extra evaluation of F). If `combined_residual_k < last_accepted_residual`:
ACCEPT w_hat, set `last_accepted_residual = combined_residual_k`. Else:
REJECT (use the plain iterate, already in place -- no write-back needed)
and CLEAR the memory. `last_accepted_residual` itself is left UNCHANGED on
a rejection (recorded design choice: the safeguard is a genuine ratchet --
AA does not fire again until the plain-ADMM trajectory has made real
progress past the last acceleration's mark; resetting the bound to +inf on
every rejection would make every subsequent single-secant candidate trivially
"pass" and defeat the ratchet's purpose).

Reject policy (P5.15 Addendum 25 item 2; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item2_aa_variant`): `anderson_acceleration['reject_policy']`, read by the
caller with a default of `DEFAULT_REJECT_POLICY` and passed to
`AndersonAccelerationState(reject_policy=...)`. The key is OPTIONAL and is not
added to `ADMMParameters`' default dict, so every existing settings dict (and
every committed run) means the default.
  - 'clear_memory' (DEFAULT; the committed Step 3.7 behaviour, run 76095561):
    a safeguard rejection takes the plain iterate, leaves the mark unchanged
    and CLEARS the memory -- exactly the paragraph above.
  - 'keep_memory' (the Addendum 25 variant): a safeguard rejection takes the
    plain iterate and leaves the mark unchanged, but RETAINS the memory,
    including this cycle's (w_k, g_k) pair (already appended before the
    decision; the deque's maxlen drops the oldest pair when full). Rationale
    (Addendum 25): every visited (w, g) pair is a valid sample of the fixed-
    point map whichever step produced it, so a rejection is no reason to
    discard the history. The record reads action 'rejected (safeguard; memory
    retained)', reset False, memory_size_after = the post-append size.
    Memory is still cleared, under BOTH policies, by `clear_for_rho_change`
    and `skip_on_failure` -- the only map changes (rho) and invalid-sample
    cycles (local solve failure). Nothing else differs between the policies.

Certificate independence (frozen spec `certificate_independence`): the
caller skips calling `AndersonAccelerationState.step` for extrapolation
(uses the plain iterate) whenever `boyd_metrics['all_boyd_pass']` is True,
so the 10 certifying cycles are plain ADMM. The (w_k, g_k) pair is still
pushed to history in that case (a recorded design choice: this is ordinary,
valid ADMM data regardless of whether an extrapolation is attempted from
it, and keeps memory "warm" so AA resumes immediately, without an empty
memory penalty, the cycle a channel leaves tolerance again).

Memory clear on rho change (frozen spec amendment (i)): the caller compares
`_update_admm_penalties`'s returned `before`/`after` per-channel rho dicts
(state, not log strings) AFTER every cycle's penalty update and calls
`AndersonAccelerationState.clear_for_rho_change` if any channel's rho
differs (a channel that is merely held/frozen/exempt this cycle has
`before[g] == after[g]` exactly -- `_scale_admm_penalty`'s `factor == 1.0`
path leaves the IEEE-754 value bit-identical -- so freezing alone never
clears memory; only an actual balancing increase/decrease, including one
that happens to coincide with an exemption lift, does).
"""

from collections import deque

import numpy as np


# P5.15 Addendum 25 item 2: the safeguard-rejection memory policy (module docstring, "Reject policy").
REJECT_POLICIES = ('clear_memory', 'keep_memory')
DEFAULT_REJECT_POLICY = 'clear_memory'


# ======================================================================================================================
#  Flag
# ======================================================================================================================
def anderson_acceleration_enabled(admm_parameters):
    settings = getattr(admm_parameters, 'anderson_acceleration', None)
    if not settings:
        return False
    return bool(settings.get('enabled', False))


# ======================================================================================================================
#  Iterate layout (static per run) and the collect/write-back round trip
# ======================================================================================================================
def build_iterate_layout(planning_problem, admm_parameters):
    """
    Build the FIXED ordering of (channel, node, year, day, period,
    power_type, kind) entries used for every w/g vector for the life of
    one `_run_operational_planning` call, together with the per-entry
    PHYSICAL scale (v_base / interface_rating / s_base_dso / 2*S_ref --
    everything except rho, which changes over cycles and is supplied fresh
    each call to `collect_w`/`write_back_w`).

    Requires `admm_parameters.shared_ess_reference_rating_mva` to be set
    (a fixed S_ref, Step 3.3(b)/3.4(d)): the ESS channel's single shared-z
    entry needs ONE scale constant, but the per-agent normalization
    `a_i = 1/(2*S_i)` (`get_admm_boyd_residual_metrics`,
    `_shared_ess_admm_normalization_mva`) is per-agent in general (S_i is
    each agent's OWN rating) and only collapses to a single constant when
    every call site reads the same fixed S_ref
    (`_admm_shared_ess_reference_mva`, `shared_resources_planning.py:4287`).
    Raises rather than silently picking one agent's rating when this does
    not hold.
    """
    s_ref = getattr(admm_parameters, 'shared_ess_reference_rating_mva', None)
    if s_ref is None:
        raise ValueError(
            'Anderson acceleration requires admm_parameters.shared_ess_reference_rating_mva '
            'to be set (a fixed S_ref shared by every shared-ESS ADMM normalization call '
            'site): the ESS channel iterate needs a single z-scale constant, which only '
            'holds under the fixed-S_ref oracle configuration. See WORKER_REPORT_S41_AA.md.'
        )
    if s_ref <= 0.0:
        raise ValueError('admm_parameters.shared_ess_reference_rating_mva must be positive.')
    ess_z_scale = 2.0 * float(s_ref)

    entries = []
    for node_id in planning_problem.active_distribution_network_nodes:
        for year in planning_problem.years:
            for day in planning_problem.days:
                network = planning_problem.transmission_network.network[year][day]
                dso_network = planning_problem.distribution_networks[node_id].network[year][day]
                v_base = network.get_node_base_kv(node_id)
                s_base_dso = dso_network.baseMVA
                interface_rating = dso_network.get_interface_branch_rating()
                for p in range(planning_problem.num_instants):
                    entries.append({
                        'channel': 'v', 'kind': 'z', 'node_id': node_id, 'year': year,
                        'day': day, 'p': p, 'power_type': None, 'scale': v_base,
                    })
                    entries.append({
                        'channel': 'v', 'kind': 'u', 'node_id': node_id, 'year': year,
                        'day': day, 'p': p, 'power_type': None, 'scale': v_base,
                    })
                    for power_type in ('p', 'q'):
                        entries.append({
                            'channel': 'pf', 'kind': 'z', 'node_id': node_id, 'year': year,
                            'day': day, 'p': p, 'power_type': power_type, 'scale': interface_rating,
                        })
                        entries.append({
                            'channel': 'pf', 'kind': 'u', 'node_id': node_id, 'year': year,
                            'day': day, 'p': p, 'power_type': power_type, 'scale': s_base_dso,
                        })
                        entries.append({
                            'channel': 'ess', 'kind': 'z', 'node_id': node_id, 'year': year,
                            'day': day, 'p': p, 'power_type': power_type, 'scale': ess_z_scale,
                        })
                        for agent in ('tso', 'dso', 'esso'):
                            entries.append({
                                'channel': 'ess', 'kind': f'u_{agent}', 'node_id': node_id,
                                'year': year, 'day': day, 'p': p, 'power_type': power_type,
                                'scale': 1.0,
                            })
    return {'entries': entries, 'n': len(entries), 'shared_ess_reference_rating_mva': float(s_ref)}


def _antisymmetry_check(tso_value, dso_value, scale):
    tol = 1e-6 * max(abs(dso_value), abs(tso_value), 1e-9) * max(scale, 1.0)
    if abs(tso_value + dso_value) > tol:
        raise ValueError(
            'Anderson acceleration: the TSO and DSO stored duals are expected to be exact '
            f'negatives of each other (rho shared across agents); got tso={tso_value!r}, '
            f'dso={dso_value!r} (sum={tso_value + dso_value!r}, tol={tol!r}). The single-u-per-'
            'channel construction is not valid under whatever state produced this.'
        )


def collect_w(layout, consensus_vars, dual_vars, rho_channel, check_antisymmetry=True):
    """
    Build the current flat w = (z, u) vector, in the FIXED order of
    `layout['entries']`, from the production consensus/dual stores. Pure
    read -- no model access, no solves.
    """
    entries = layout['entries']
    w = np.empty(len(entries), dtype=float)
    for i, e in enumerate(entries):
        node_id, year, day, p, pt = e['node_id'], e['year'], e['day'], e['p'], e['power_type']
        channel, kind, scale = e['channel'], e['kind'], e['scale']
        if channel == 'v':
            if kind == 'z':
                raw = consensus_vars['vmag']['tso']['current'][node_id][year][day][p]
                w[i] = raw / scale
            else:
                dso_raw = dual_vars['vmag']['dso']['current'][node_id][year][day][p]
                if check_antisymmetry:
                    tso_raw = dual_vars['vmag']['tso']['current'][node_id][year][day][p]
                    _antisymmetry_check(tso_raw, dso_raw, scale)
                w[i] = (dso_raw / scale) / rho_channel['v']
        elif channel == 'pf':
            if kind == 'z':
                raw = consensus_vars['pf']['tso']['current'][node_id][year][day][pt][p]
                w[i] = raw / scale
            else:
                dso_raw = dual_vars['pf']['dso']['current'][node_id][year][day][pt][p]
                if check_antisymmetry:
                    tso_raw = dual_vars['pf']['tso']['current'][node_id][year][day][pt][p]
                    _antisymmetry_check(tso_raw, dso_raw, scale)
                w[i] = (dso_raw / scale) / rho_channel['pf']
        else:
            if kind == 'z':
                raw = consensus_vars['ess']['z']['current'][node_id][year][day][pt][p]
                w[i] = raw / scale
            else:
                agent = kind.split('_', 1)[1]
                raw = dual_vars['ess'][agent]['current'][node_id][year][day][pt][p]
                w[i] = (raw / scale) / rho_channel['ess']
    return w


def write_back_w(layout, w, consensus_vars, dual_vars, rho_channel):
    """
    Inverse of `collect_w`: write an (accepted, extrapolated) w vector back
    into the SAME stores `collect_w` reads, plus the TSO's mirrored dual
    copy for V/PF (see module docstring: dual_vars[...]['tso']['current']
    == -dual_vars[...]['dso']['current'] exactly, an invariant of the
    plain update this function restores explicitly rather than relying on
    it persisting through an AA step).

    Proximal centres (`prox_v_prev`/`prox_pf_prev`/`prox_ess_prev` on the
    TSO model) are NOT written here: under the oracle configuration
    (`proximal_regularization.tso.tau == 0.0`, `gamma_policy ==
    'tied_to_rho'` => `prox_gamma_* = tau*rho = 0`), the proximal term
    `(gamma/2)*(x - prox_prev)**2` is multiplied by gamma == 0 and is
    inert regardless of `prox_prev`'s value -- verified in
    `update_transmission_model_to_admm` (the term is only ever added
    scaled by `prox_gamma_*`). A run with tau != 0 would need this
    reconsidered; not required by the oracle this module targets.
    """
    entries = layout['entries']
    for i, e in enumerate(entries):
        node_id, year, day, p, pt = e['node_id'], e['year'], e['day'], e['p'], e['power_type']
        channel, kind, scale = e['channel'], e['kind'], e['scale']
        value = float(w[i])
        if channel == 'v':
            if kind == 'z':
                consensus_vars['vmag']['tso']['current'][node_id][year][day][p] = value * scale
            else:
                y = value * rho_channel['v'] * scale
                dual_vars['vmag']['dso']['current'][node_id][year][day][p] = y
                dual_vars['vmag']['tso']['current'][node_id][year][day][p] = -y
        elif channel == 'pf':
            if kind == 'z':
                consensus_vars['pf']['tso']['current'][node_id][year][day][pt][p] = value * scale
            else:
                y = value * rho_channel['pf'] * scale
                dual_vars['pf']['dso']['current'][node_id][year][day][pt][p] = y
                dual_vars['pf']['tso']['current'][node_id][year][day][pt][p] = -y
        else:
            if kind == 'z':
                consensus_vars['ess']['z']['current'][node_id][year][day][pt][p] = value * scale
            else:
                agent = kind.split('_', 1)[1]
                dual_vars['ess'][agent]['current'][node_id][year][day][pt][p] = value * rho_channel['ess']


# ======================================================================================================================
#  Type-II AA state machine
# ======================================================================================================================
class AndersonAccelerationState:
    """
    One instance per `_run_operational_planning` call (mirrors
    `freeze_state`/`tso_pristine_base`'s lifetime -- constructed once
    before the ADMM loop, mutated in place every cycle). See the module
    docstring for the algorithm and the safeguard's exact semantics.
    """

    def __init__(self, memory=5, regularization=1e-10, reject_policy=DEFAULT_REJECT_POLICY):
        self.memory = int(memory)
        if self.memory < 1:
            raise ValueError('Anderson acceleration memory must be at least 1.')
        self.regularization = float(regularization)
        if reject_policy not in REJECT_POLICIES:
            raise ValueError(f'Anderson acceleration reject_policy must be one of {REJECT_POLICIES}; '
                             f'got {reject_policy!r}.')
        self.reject_policy = reject_policy
        self._history = deque(maxlen=self.memory + 1)  # (w, g) pairs, oldest first
        self.last_accepted_residual = float('inf')
        self.records = []

    def memory_size(self):
        """Number of usable Delta-columns (secant pairs) right now."""
        return max(len(self._history) - 1, 0)

    def clear_memory(self):
        self._history.clear()

    def clear_for_rho_change(self, cycle, channels_changed):
        memory_before = self.memory_size()
        self.clear_memory()
        record = {
            'cycle': cycle,
            'action': 'memory reset (rho change)',
            'accepted': False,
            'combined_residual': None,
            'baseline_residual_before': self.last_accepted_residual,
            'memory_size_before': memory_before,
            'memory_size_after': 0,
            'reset': True,
            'reset_reason': f'rho changed on channel(s): {list(channels_changed)}',
            'rho_changed_channels': list(channels_changed),
        }
        self.records.append(record)
        return record

    def skip_on_failure(self, cycle):
        """
        A local solve failed this cycle: consensus_vars/dual_vars were NOT
        fully updated for the failed agent(s)
        (`_update_interface_power_flow_variables`/
        `_update_shared_energy_storage_variables`, both gated on
        `_solver_result_succeeded`), so g_k = w_{k+1}-w_k would mix a
        stale sub-block with fresh ones and is not a valid secant pair.
        Recorded as a distinct reset reason; not part of the frozen spec's
        literal safeguard rule (which assumes every cycle produces a valid
        F-evaluation) but required for the AA state to stay well-defined
        across a solver-failure cycle. `last_accepted_residual` is left
        unchanged -- a failure is not progress, but it is also not
        evidence the last acceleration's mark should be abandoned.
        """
        memory_before = self.memory_size()
        self.clear_memory()
        record = {
            'cycle': cycle,
            'action': 'skipped (local solve failure this cycle)',
            'accepted': False,
            'combined_residual': None,
            'baseline_residual_before': self.last_accepted_residual,
            'memory_size_before': memory_before,
            'memory_size_after': 0,
            'reset': True,
            'reset_reason': 'local solve failure this cycle',
            'rho_changed_channels': [],
        }
        self.records.append(record)
        return record

    def step(self, cycle, w_k, g_k, combined_residual, boyd_all_pass):
        """
        One cycle's AA decision. Returns (w_next, record). `w_next` is the
        vector the caller should ensure is what the stores hold for the
        next cycle: either the plain iterate (already there -- caller does
        nothing) or the accepted extrapolation (caller must write it back
        via `write_back_w`); `record['action'] == 'accepted'` distinguishes
        the two, `record['accepted']` is the same flag as a bool.
        """
        memory_before = self.memory_size()
        w_plain = w_k + g_k

        record = {
            'cycle': cycle,
            'combined_residual': float(combined_residual),
            'baseline_residual_before': self.last_accepted_residual,
            'memory_size_before': memory_before,
            'boyd_all_pass': bool(boyd_all_pass),
            'rho_changed_channels': [],
        }

        if boyd_all_pass:
            # Certificate independence: no extrapolation attempted while
            # every channel is inside its Boyd tolerance. Still push the
            # plain (w_k, g_k) pair -- ordinary, valid ADMM data -- so AA
            # resumes with a warm memory the cycle a channel leaves
            # tolerance again (recorded design choice, module docstring).
            self._history.append((w_k, g_k))
            record.update(
                action='off (all channels within Boyd tolerance)',
                accepted=False,
                memory_size_after=self.memory_size(),
                reset=False, reset_reason=None,
            )
            self.records.append(record)
            return w_plain, record

        self._history.append((w_k, g_k))
        m_k = self.memory_size()

        if m_k < 1:
            record.update(
                action='insufficient memory (m_k=0)',
                accepted=False,
                memory_size_after=self.memory_size(),
                reset=False, reset_reason=None,
            )
            self.records.append(record)
            return w_plain, record

        pairs = list(self._history)
        delta_w = np.stack([pairs[i + 1][0] - pairs[i][0] for i in range(len(pairs) - 1)], axis=1)
        delta_g = np.stack([pairs[i + 1][1] - pairs[i][1] for i in range(len(pairs) - 1)], axis=1)

        gram = delta_g.T @ delta_g
        gram = gram + self.regularization * np.eye(gram.shape[0])
        rhs = delta_g.T @ g_k
        gamma = np.linalg.solve(gram, rhs)
        w_hat = w_k + g_k - (delta_w + delta_g) @ gamma

        if combined_residual < self.last_accepted_residual:
            self.last_accepted_residual = float(combined_residual)
            record.update(
                action='accepted',
                accepted=True,
                memory_size_after=self.memory_size(),
                gamma_columns=int(m_k),
                reset=False, reset_reason=None,
            )
            self.records.append(record)
            return w_hat, record
        elif self.reject_policy == 'keep_memory':
            # Addendum 25 variant: plain iterate, mark unchanged, memory
            # RETAINED -- (w_k, g_k) was appended above and stays.
            record.update(
                action='rejected (safeguard; memory retained)',
                accepted=False,
                memory_size_after=self.memory_size(),
                gamma_columns=int(m_k),
                reset=False, reset_reason=None,
            )
            self.records.append(record)
            return w_plain, record
        else:
            self.clear_memory()
            record.update(
                action='rejected (safeguard)',
                accepted=False,
                memory_size_after=self.memory_size(),
                gamma_columns=int(m_k),
                reset=True, reset_reason='safeguard rejection (combined residual did not fall below the last accepted mark)',
            )
            self.records.append(record)
            return w_plain, record


def combined_scaled_residual(boyd_metrics):
    """
    The "combined scaled residual" the safeguard compares
    (`item4_step_3_7_anderson.method.safeguard`): Euclidean combination of
    the Boyd primal (r) and dual (s) residual norms of all three channels,
    already computed by `get_admm_boyd_residual_metrics` every cycle
    regardless of AA -- no extra evaluation of F. Recorded design choice
    (module docstring).
    """
    total = 0.0
    for group in ('v', 'pf', 'ess'):
        total += boyd_metrics[group]['r'] ** 2 + boyd_metrics[group]['s'] ** 2
    return total ** 0.5
