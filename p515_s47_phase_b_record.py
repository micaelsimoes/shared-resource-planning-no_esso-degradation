"""
P5.15 Addendum 30 (task W23) -- Phase B FORMAL RECORD under the ageing BASELINE (spec v17 step S4):
the STEP4_DFO_METHOD.md section 5 MADS poll from the incumbent, under the budget B = 1e6 EUR, with every
committed certified BASELINE evaluation as the cache, through the campaign harness
(`p515_s44_campaign_harness.py`, unchanged) with the SAME declared `ess_ageing_baseline` as S2/S3
(`p515_s47_baseline_campaign.py`, 6642b33b; harness declaration 65525006), so the eval key of a candidate
here equals its S2/S3 eval key and the S2/S3 records ARE the cache.

AUTHORITY: PLANNER_BRIEF_2026-09-13.md Addenda 27 (budget, single-cohort Phase B first, <= 10 directions per
poll), 28 ("Phase B under the current baseline: run as the formal record only"), 29 (concurrency 5), 30
("Phase B: formal record under the chosen baseline, after the ladders"); STEP4_DFO_METHOD.md sections 1.2,
1.3, 2.4, 2.6, 3, 5.2-5.5, 6, 8; frozen spec v17 step S4 and its prediction.

================================================================================
WHAT IS IMPLEMENTED (STEP4 line references are to STEP4_DFO_METHOD.md as read for W23)
================================================================================
Decision space (STEP4 1.1 l.25-28, 5.1 l.183-185; Addenda 27/28). Single-cohort form restricted to ONE common
investment year: z = (zP5, zE5, zP7, zE7, zP9, zE9, zY), n = 7, P = 0.25 zP MVA, E = 0.5 zE MWh (STEP4 3
l.121-122: z = D^-1 x, D = diag(0.25, 0.5, ...)), zY = index into the instance years (2025, 2030, 2035; the
ordinal "5-year timing" step, STEP4 8 l.277). The harness's canonical candidate carries ONE investment year
for the whole candidate (`canonical_candidate`; multi-cohort not supported), so per-node years (the 9-variable
form of STEP4 1.1 l.25-26) are NOT evaluable by the frozen oracle -- see the worker report (ambiguity A1).

Constraints (STEP4 1.2 l.30-40), closed form, before any evaluation (5.5 l.214): zP, zE >= 0; P = 0 <=> E = 0;
2P <= E <= 4P  <=>  zP <= zE <= 2 zP; E <= 5 MWh  <=>  zE <= 10; zY in {0, 1, 2}; I(x) <= B = 1e6.
I(x) (STEP4 1.3 l.54-58) = sum_n c^S_y P_n + c^E_y E_n with the discounted, weight-expected unit costs of the
pinned W2 table (`investment_cost_results.json` expected_unit_costs_per_case_year.new, corrected cost file
e17bd588...): linear, identical to production's `model.investment_cost` -- re-checked against all 105 W2
candidates at --freeze / --run (max abs diff <= 1e-6 EUR).

Canonicalization (STEP4 5.4 l.207-210): P = 0 entries carry E = 0; with no storage at any node the year is
inactive and set to y1 = 2025; a poll point whose canonical form equals the incumbent's is DROPPED (a
direction that only changes inactive entries). Identity = the harness eval key under the declared baseline.

Poll (STEP4 3 l.117-118, 5.3 l.199-202; Addendum 27 "<= 10 directions per poll"): OrthoMADS n+1 = 8
directions. v = 2 u_t - 1 normalized, u_t the Halton point of index t = HALTON_T0 + k (bases = the first n
primes; t0 = p_n = 17, Abramson et al. 2009; k = poll counter); H = I - 2 v v^T; directions h_1..h_n (the
columns of H) and h_{n+1} = -sum_j h_j (the "n+1 NEG" completion, Audet-Ianni-Le Digabel-Tribes 2014); each
direction projected on the unit lattice at poll size Delta as d = round(Delta h / ||h||_inf) (round half away
from zero), so ||d||_inf = Delta (NOMAD 4 GMesh scaleAndProjectOnMesh with mesh size 1). FULL POLL: every
direction is evaluated (no opportunistic stop), in batches of <= CONCURRENCY = 5; the incumbent is updated after
the whole poll (5.3 l.200-201). Cache hits are never re-evaluated (2.6 l.101).

Mesh / poll size (STEP4 5.2 l.189-195): Delta_0 = 4 scaled units; Delta doubles on success and halves on
failure, never below 1; the mesh size is 1 lattice unit in every coordinate (the granularity floor: for
poll sizes below 10 granules NOMAD 4's granular mesh size G max(1, 10^(b - |b - b0|)) equals G). sigma_Q
(STEP4 4.3 l.173-174, 1.4 l.72-73; Addendum 28): the measured value pinned from the committed Phase A tables
(T3 residual_max_abs_eur = 18,449.66 EUR, the upper end of Addendum 28's "10-18k"); 5.2's degradation clause
(sigma_Q > one lattice step's investment cost) is evaluated in closed form at --freeze and recorded; it does not
trigger (min unit-step cost of a P or E coordinate > 2 sigma_Q). Delta_0 is therefore 4 (5.2 l.189).

Success / the indeterminate rule (STEP4 3 l.118-125 "none improves F"; 8 l.278-281; CLAUDE.md "report a
difference with its resolution"): a polled point IMPROVES the incumbent only if
    F(inc) - F(x) > resolution(x, inc) = max(bar(x) + bar(inc), sigma_Q)
(bar = the record's max gross step over the last 10 cycles). 0 < F(inc) - F(x) <= resolution is recorded as
INDETERMINATE and NOT accepted. The poll succeeds iff at least one determinate improvement exists; the new
incumbent is the argmin F among the determinate improvers (ties: lower I, then label). STEP4 does not state
this threshold explicitly (ambiguity A4 in the report); section 8 l.278-280 ("or, if the poll stopped above
unit size because of sigma_Q, ... the finer neighbourhood reported as unresolved") is followed by listing every
indeterminate point of the final poll as UNRESOLVED in the terminal record.

Extreme barrier (STEP4 2.4 l.92-96, 5.5 l.214-215): an infeasible poll point is rejected before evaluation
(F = +inf at zero cost, with its reasons); a non-certified evaluation (any status other than 'certified') is
F = +inf, recorded with its cause, and the direction counts as evaluated (records carry F_eur = None with
outcome 'barrier' / 'barrier_infeasible_not_evaluated', and a missing / infinite resolution as None: strict JSON). A run of barrier points stops the
campaign for review (2.4 l.95-96): STOP after a batch if >= 2 NEW barrier evaluations in the current poll or
>= 3 overall (a1a's stop rule, transcribed; ambiguity A7).

Termination (STEP4 3 l.123-126): the poll at unit poll size (Delta = 1) fails -> mesh-local optimum; or the
evaluation budget MAX_NEW_EVALUATIONS is exhausted (a poll that would exceed it is not launched; the incumbent
is reported with the poll size reached); or MAX_POLLS (safety); or the barrier stop rule.

Initial incumbent (task W23): argmin F over the budget-feasible, on-lattice, CERTIFIED cache entries plus x = 0
(ties: lower I, then label). x = 0 is NOT re-run: Q(0) is pinned from the A0 x0 record (ageing-independent,
W21's evidence 2466401d, re-checked here exactly as the S2/S3 launcher does). The margin of the chosen incumbent
over F(0) and over the runner-up is recorded with its resolution.

Cache (STEP4 2.6 l.101-102, 5.4): every committed BASELINE campaign result under data/SRP1/Results/P515S47/
(`campaign_*/campaign_results.json`, git-tracked and clean) whose frozen spec declares the SAME ageing baseline,
case-file AA, ESS params sha256, case-file sha256, cap 500, 10 cycles, arm s39_D, no overrides and no model
variant; every point's eval key is recomputed and must match. C3-era results (P515S44/S45/S46) are NOT cache
(different model): asserted by scanning them -- no C3 eval key may equal any cache or domain eval key and no C3
spec may declare the baseline. Duplicate eval keys (S2 and S3 both hold n7_4h_e1) must agree bitwise on status,
Q and bar, else the freeze refuses. The frozen spec pins every cache source by sha256 and freezes the cache
table; --run recomputes both and refuses on any difference.

Harness use: the frozen spec's `candidates` are the WHOLE admissible domain (every budget-, bound- and
duration-feasible common-year lattice point, x = 0 first; 1,705 entries on SRP1), so the harness's
`evaluate(labels, ctx)` can run any polled point under ONE frozen spec and ONE campaign lock (STEP4 6 l.244-246);
only polled, non-cached points are ever evaluated.

Resumability (STEP4 6 l.244): the state (poll history, incumbent, Delta, cache additions) is rewritten
atomically to `phase_b_state.json` after every poll. The algorithm is deterministic given the cache and Q is
deterministic, so a re-run (new root, `--campaign-id s47_phase_b_rN`) after committing this root's
campaign_results.json retraces the same polls with every evaluated point a cache hit.

ZERO SOLVES in the parent: SolveProfileGuard(permitted=()) installed at import, verified at exactly 0.

EXACT COMMANDS (repo root, canonical interpreter, attached, both streams captured, never detached; only after
S3 is finished and committed):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_phase_b_record.py --freeze \\
      > data/SRP1/Results/P515S47/campaign_s47_phase_b_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_phase_b_record.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S47/campaign_s47_phase_b_launch.log 2>&1
Exit codes (--run): 0 terminated by the unit-poll failure; 2 evaluation budget / MAX_POLLS reached; 3
STOP_FOR_REVIEW (barrier stop rule); 1 harness / guard / precondition failure.
"""

import argparse
import ast
import glob
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S47 Phase B record parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402  (stdlib only at import; the parent imports no model code)

LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
STAGE = 'P5.15 Addendum 30 S4 -- Phase B formal record under the ageing BASELINE (STEP4 section 5 MADS poll, B = 1e6)'
_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
DEFAULT_CAMPAIGN_ID = 's47_phase_b'
CAMPAIGN_ID_PATTERN = re.compile(r'^s47_phase_b(_r[0-9]+)?$')

# ---- the frozen configuration: IDENTICAL to p515_s47_baseline_campaign.py (6642b33b) ----
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
ARM_LABEL = 's39_D'

# ---- the master problem (STEP4 1.1-1.3; spec v15 master_constraints) ----
ACTIVE_NODES = (5, 7, 9)
GRANULE_P_MVA = 0.25
GRANULE_E_MWH = 0.5
E_MAX_MWH = 5.0
ZE_MAX = 10  # E_MAX_MWH / GRANULE_E_MWH
BUDGET_EUR = 1e6
N_VARS = 2 * len(ACTIVE_NODES) + 1  # 7: (zP, zE) per node + the common year index

# ---- the poll design (STEP4 3, 5.2, 5.3) ----
DELTA_0 = 4
DELTA_MIN = 1
MESH_SIZE = 1
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
HALTON_T0 = PRIMES[N_VARS - 1]  # t0 = p_n (Abramson, Audet, Dennis, Le Digabel 2009)
POLL_DESIGN = 'orthomads_n_plus_1_neg'
MAX_NEW_EVALUATIONS = 20
MAX_POLLS = 60
BARRIER_STOP_PER_POLL = 2
BARRIER_STOP_OVERALL = 3

# ---- pins ----
SPEC_V17 = {'path': os.path.join(_P47, 'frozen_s47_baseline_spec_v17_ff0056b8.json'),
            'sha256': 'ff0056b850957d47cc3aa2595eab13497e100cc7bcda84514c5ca58e25d1e109'}
SPEC_V15 = {'path': os.path.join(_P45, 'frozen_s45_phaseA_spec_v15_5feefd7b.json'),
            'sha256': '5feefd7b642fc3d480156ad5e52ed6e1cf9d6698cfd40dbb33389bab6e6229fe'}
ESS_PARAMS_FILE = {'path': H.ESS_PARAMS_FILE_REL,
                   'sha256': '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706', 'commit': '2466401d'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
A0_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json'),
              'sha256': '423678b98c0ca9a26dbc76a1b46a5e36a51c8ccccc0459253020bfe1061653b4', 'label': 'x0'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'field': 'expected_unit_costs_per_case_year.new.<year>.{power_eur_per_mva_discounted, '
                                    'energy_eur_per_mwh_discounted}; candidates.<label>.I_new_eur (cross-check)'}
PHASE_A_TABLES = {'path': os.path.join(_P45, 'phase_a_tables', 'phase_a_tables.json'),
                  'sha256': 'a2b78541b5c50ad2a185ca3eb3d0f1b38bb388c4866c45628a984c407b4a3995', 'commit': 'c1b64278',
                  'field': 'T3.residual_max_abs_eur (sigma_Q; T3.residual_rms_eur recorded beside it)'}
BASELINE_LAUNCHER = {'path': 'p515_s47_baseline_campaign.py', 'commit': '6642b33b'}
IDENTITY_CHECKS = {'path': os.path.join(_P47, 'identity_checks', 'identity_checks.json'), 'commit': '65525006',
                   'expect': 'all_ok true'}
CASE_FILE_GATE = {'path': os.path.join(_P47, 'case_file_baseline', 'after', 'case_file_baseline_after.json'),
                  'commit': '2466401d', 'expect': 'all_ok false, only b_x0_every_block_digest_identical_before_after'}
X0_ANALYSIS = {'path': os.path.join(_P47, 'case_file_baseline', 'x0_diff_analysis', 'x0_diff_analysis.json'),
               'commit': '2466401d', 'expect': 'all_ok true'}
GATE_EXPECTED_FAILING = ['b_x0_every_block_digest_identical_before_after']

CACHE_SOURCE_GLOB = os.path.join(_P47, 'campaign_*', 'campaign_results.json')
C3_ERA_GLOBS = (os.path.join('data', 'SRP1', 'Results', 'P515S44', '**', 'campaign_results.json'),
                os.path.join('data', 'SRP1', 'Results', 'P515S45', '**', 'campaign_results.json'),
                os.path.join('data', 'SRP1', 'Results', 'P515S46', '**', 'campaign_results.json'))

OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); F(x) = I(x) + Q(x); '
                        'I(0) = 0 so F(0) = Q(0), Q(0) from the pinned A0 x0 record; salvage and net recourse '
                        'reported by the harness records, excluded from F (Addendum 27 item 3).')
RESOLUTION_RULE = ('a polled x improves the incumbent iff F(inc) - F(x) > max(bar(x) + bar(inc), sigma_Q); '
                   '0 < F(inc) - F(x) <= that resolution is INDETERMINATE and not accepted; bar = record max gross '
                   'step over the last 10 cycles; a missing bar makes the resolution +inf')
STOP_RULE = (f'STOP_FOR_REVIEW (no further batch) if >= {BARRIER_STOP_PER_POLL} NEW barrier evaluations in one poll '
             f'or >= {BARRIER_STOP_OVERALL} overall (STEP4 2.4 "a run of barrier points in a region stops the '
             'campaign for review"; a1a stop-rule thresholds)')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addenda 27, 28, 29, 30 (Phase B formal record under the chosen baseline)',
    'STEP4_DFO_METHOD.md sections 1.2, 1.3, 2.4, 2.6, 3, 5.2-5.5, 6, 8',
    'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json step S4',
    'Planner task W23 (launcher; concurrency 5, cap 500, 10 cycles, AA-on case file, baseline declared)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py')

POLL_RECORD_FIELDS = ('poll_index', 'halton_t', 'poll_size_delta', 'mesh_size', 'incumbent', 'directions',
                      'candidates', 'n_new_evaluations', 'n_cache_hits', 'batches', 'decision', 'next_incumbent',
                      'next_poll_size', 'unit_poll')
CANDIDATE_RECORD_FIELDS = ('direction_index', 'direction', 'z', 'label', 'eval_key', 'canonical', 'feasible',
                           'infeasibility_reasons', 'I_x_eur', 'disposition', 'source', 'status', 'barrier_cause',
                           'Q_eur', 'F_eur', 'bar_eur', 'incumbent_bar_eur', 'bar_sum_eur', 'sigma_Q_eur',
                           'resolution_eur', 'F_inc_minus_F_eur', 'outcome')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


# ======================================================================================================================
#  the lattice (pure)
# ======================================================================================================================
class Lattice:
    """The common-year single-cohort lattice. z = [zP5, zE5, zP7, zE7, zP9, zE9, zY] (ints)."""

    def __init__(self, years, unit_costs, budget=BUDGET_EUR, nodes=ACTIVE_NODES):
        self.years = tuple(int(y) for y in years)
        self.nodes = tuple(nodes)
        self.budget = float(budget)
        self.c_p = {int(y): float(unit_costs[int(y)]['power']) for y in self.years}
        self.c_e = {int(y): float(unit_costs[int(y)]['energy']) for y in self.years}
        if len(self.nodes) * 2 + 1 != N_VARS:
            raise ValueError('lattice dimension mismatch')

    @staticmethod
    def x0():
        return (0,) * N_VARS

    def node_pairs(self, z):
        return [(z[2 * i], z[2 * i + 1]) for i in range(len(self.nodes))]

    def has_storage(self, z):
        return any(p != 0 or e != 0 for p, e in self.node_pairs(z))

    def reasons(self, z):
        """Every violated constraint of STEP4 1.2 (empty list = feasible). Order: bounds, P=0<=>E=0, duration,
        E_max, year bounds (only when storage is present: otherwise the year is inactive, 5.4), budget."""
        out = []
        for node, (zp, ze) in zip(self.nodes, self.node_pairs(z)):
            if zp < 0 or ze < 0:
                out.append(f'bound: node {node} negative capacity (P={zp * GRANULE_P_MVA:g}, E={ze * GRANULE_E_MWH:g})')
                continue
            if (zp == 0) != (ze == 0):
                out.append(f'duration: node {node} P = 0 <=> E = 0 violated (P={zp * GRANULE_P_MVA:g}, '
                           f'E={ze * GRANULE_E_MWH:g})')
                continue
            if zp > 0 and not (zp <= ze <= 2 * zp):
                out.append(f'duration: node {node} E/P = {ze * GRANULE_E_MWH / (zp * GRANULE_P_MVA):g} h outside [2, 4] h')
            if ze > ZE_MAX:
                out.append(f'bound: node {node} E = {ze * GRANULE_E_MWH:g} MWh > {E_MAX_MWH:g} MWh')
        if self.has_storage(z) and not (0 <= z[-1] < len(self.years)):
            out.append(f'bound: year index {z[-1]} outside 0..{len(self.years) - 1}')
        if not out and self.investment_cost(z) > self.budget:
            out.append(f'budget: I(x) = {self.investment_cost(z):.2f} > B = {self.budget:g}')
        return out

    def canonical_z(self, z):
        """STEP4 5.4: no storage -> the year is inactive and set to y1 (index 0)."""
        z = tuple(int(v) for v in z)
        if not self.has_storage(z):
            return self.x0()
        return z

    def year_of(self, z):
        return self.years[self.canonical_z(z)[-1]]

    def investment_cost(self, z):
        if not self.has_storage(z):
            return 0.0
        y = self.years[z[-1]]
        return sum(self.c_p[y] * GRANULE_P_MVA * zp + self.c_e[y] * GRANULE_E_MWH * ze for zp, ze in self.node_pairs(z))

    def nodes_map(self, z):
        return {n: (GRANULE_P_MVA * zp, GRANULE_E_MWH * ze) for n, (zp, ze) in zip(self.nodes, self.node_pairs(z))}

    def label(self, z):
        z = self.canonical_z(z)
        if not self.has_storage(z):
            return 'x0'
        parts = [f'n{n}_p{GRANULE_P_MVA * zp:g}_e{GRANULE_E_MWH * ze:g}'
                 for n, (zp, ze) in zip(self.nodes, self.node_pairs(z)) if zp or ze]
        return f'y{self.years[z[-1]]}__' + '__'.join(parts)

    def z_of_canonical(self, canonical):
        """Inverse map from a harness canonical candidate; None when off-lattice or year not an instance year."""
        year = int(canonical['investment_year'])
        z = []
        for n in self.nodes:
            s_val, e_val = canonical['nodes'][str(n)]
            zp, ze = s_val / GRANULE_P_MVA, e_val / GRANULE_E_MWH
            if zp != int(zp) or ze != int(ze):
                return None
            z += [int(zp), int(ze)]
        if year not in self.years:
            return None
        z.append(self.years.index(year))
        z = tuple(z)
        if not self.has_storage(z) and year != self.years[0]:
            return None  # a non-canonical x = 0 (year must be y1)
        return z

    def domain(self):
        """Every feasible canonical point, x = 0 first, then by (year, z)."""
        pts = [self.x0()]
        per_node = [(zp, ze) for zp in range(0, ZE_MAX + 1) for ze in range(0, ZE_MAX + 1)
                    if (zp == 0 and ze == 0) or (zp > 0 and zp <= ze <= 2 * zp)]
        for yi in range(len(self.years)):
            for combo in product(per_node, repeat=len(self.nodes)):
                z = tuple(v for pair in combo for v in pair) + (yi,)
                if self.has_storage(z) and not self.reasons(z):
                    pts.append(z)
        return pts

    def neighbourhood(self, z_inc):
        """Every feasible canonical lattice point x' != x with ||z' - z||_inf <= 1 (STEP4 8 'lattice neighbour'),
        independently of the polled directions -- a RECORD of what the unit poll covered, never evaluated."""
        seen, out = set(), []
        for d in product((-1, 0, 1), repeat=N_VARS):
            z = self.canonical_z(tuple(a + b for a, b in zip(z_inc, d)))
            if z == self.canonical_z(z_inc) or z in seen:
                continue
            if not self.reasons(tuple(a + b for a, b in zip(z_inc, d))):
                seen.add(z)
                out.append(z)
        return sorted(out)


# ======================================================================================================================
#  OrthoMADS directions (pure)
# ======================================================================================================================
def radical_inverse(t, base):
    f, r = 1.0, 0.0
    while t > 0:
        f /= base
        r += f * (t % base)
        t //= base
    return r


def halton_point(t, n=N_VARS):
    return [radical_inverse(t, PRIMES[i]) for i in range(n)]


def _round_half_away(x):
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def householder_columns(t, n=N_VARS):
    u = halton_point(t, n)
    v = [2.0 * a - 1.0 for a in u]
    norm = math.sqrt(sum(a * a for a in v))
    if norm == 0.0:
        raise ValueError(f'degenerate Halton direction at t={t}')
    v = [a / norm for a in v]
    return u, [[(1.0 if i == j else 0.0) - 2.0 * v[i] * v[j] for i in range(n)] for j in range(n)]


def project_direction(h, delta):
    m = max(abs(a) for a in h)
    return tuple(_round_half_away(delta * a / m) for a in h)


def poll_directions(k, delta, n=N_VARS, t0=HALTON_T0, design=POLL_DESIGN):
    """OrthoMADS n+1 NEG on the unit lattice: H = I - 2 v v^T from the Halton point t = t0 + k; directions =
    the n columns and -sum of the columns, each rounded to ||d||_inf = delta. Returns (t, u, directions)."""
    if design != 'orthomads_n_plus_1_neg':
        raise ValueError(f'unsupported poll design {design}')
    t = t0 + k
    u, cols = householder_columns(t, n)
    neg = [-sum(c[i] for c in cols) for i in range(n)]
    return t, u, [project_direction(c, delta) for c in cols] + [project_direction(neg, delta)]


# ======================================================================================================================
#  decisions (pure)
# ======================================================================================================================
def resolution(bar_x, bar_inc, sigma_q):
    if bar_x is None or bar_inc is None:
        return float('inf')
    return max(bar_x + bar_inc, sigma_q)


def classify(f_inc, f_x, res):
    """'barrier' (F = inf), 'improvement' (> res), 'indeterminate' (0 < diff <= res), 'no_improvement'."""
    if f_x is None or not math.isfinite(f_x):
        return 'barrier'
    diff = f_inc - f_x
    if diff > res:
        return 'improvement'
    if diff > 0:
        return 'indeterminate'
    return 'no_improvement'


def _order_key(entry):
    return (entry['F'], entry['I'], entry['label'])


def initial_incumbent(lattice, cache, x0_eval_key, sigma_q):
    """argmin F over certified, on-lattice, budget-feasible cache entries (x = 0 included; ties -> lower I, then
    label). Returns (incumbent entry, record)."""
    eligible, excluded = [], []
    for ekey, c in sorted(cache.items()):
        z = lattice.z_of_canonical(c['canonical'])
        if z is None:
            excluded.append({'eval_key': ekey, 'label': c.get('label'), 'reason': 'off the lattice / not an instance year'})
            continue
        why = lattice.reasons(z)
        if why:
            excluded.append({'eval_key': ekey, 'label': c.get('label'), 'reason': '; '.join(why)})
            continue
        if c['status'] != 'certified' or c.get('Q') is None:
            excluded.append({'eval_key': ekey, 'label': c.get('label'), 'reason': f"barrier ({c['status']})"})
            continue
        i_x = lattice.investment_cost(z)
        eligible.append({'eval_key': ekey, 'z': z, 'label': lattice.label(z), 'I': i_x, 'Q': c['Q'],
                         'F': i_x + c['Q'], 'bar': c.get('bar'), 'source': c.get('source')})
    if x0_eval_key not in {e['eval_key'] for e in eligible}:
        raise AssertionError('x = 0 must be an eligible (certified, pinned) cache entry')
    eligible.sort(key=_order_key)
    inc = eligible[0]
    x0 = next(e for e in eligible if e['eval_key'] == x0_eval_key)
    runner = eligible[1] if len(eligible) > 1 else None

    def _margin(other):
        if other is None:
            return None
        res = resolution(inc['bar'], other['bar'], sigma_q)
        diff = other['F'] - inc['F']
        return {'other': other['label'], 'F_other_minus_F_incumbent_eur': diff,
                'resolution_eur': res if math.isfinite(res) else None,
                'determinate': diff > res}

    return inc, {'rule': 'argmin F over certified, on-lattice, budget-feasible cache entries incl. x = 0; ties -> '
                         'lower I, then label',
                 'incumbent': {k: inc[k] for k in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar', 'source')},
                 'is_x0': inc['eval_key'] == x0_eval_key,
                 'margin_vs_x0': None if inc['eval_key'] == x0_eval_key else _margin(x0),
                 'margin_vs_runner_up': _margin(runner),
                 'every_storage_point_F_gt_F0': all(e['F'] > x0['F'] for e in eligible if e['eval_key'] != x0_eval_key),
                 'n_eligible': len(eligible),
                 'eligible_ranked': [{k: e[k] for k in ('label', 'F', 'I', 'bar')} for e in eligible],
                 'excluded': excluded}


def run_mads(lattice, cache, key_of, incumbent, evaluate_fn, sigma_q, delta0=DELTA_0,
             max_new_evaluations=MAX_NEW_EVALUATIONS, max_polls=MAX_POLLS, batch_size=CONCURRENCY,
             log=_log, on_poll=None):
    """The Phase B loop. `cache`: eval_key -> {'status', 'Q', 'bar', 'canonical', 'source', ...} (extended in place
    with new evaluations). `key_of(z)` -> eval key of the canonical z. `evaluate_fn(list[z]) -> list[cache entry]`
    (<= batch_size per call). Returns the terminal record."""
    inc = dict(incumbent)
    delta = int(delta0)
    history, n_new, n_barrier_new = [], 0, 0
    termination = None

    def _finish(rec):
        history.append(rec)
        if on_poll:
            on_poll(rec)

    for k in range(max_polls):
        t, u, dirs = poll_directions(k, delta)
        inc_view = {kk: inc[kk] for kk in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar')}
        cands, seen = [], {}
        for j, d in enumerate(dirs):
            raw = tuple(a + b for a, b in zip(inc['z'], d))
            why = lattice.reasons(raw)
            entry = {'direction_index': j, 'direction': list(d), 'z': list(raw), 'label': None, 'eval_key': None,
                     'canonical': None, 'feasible': not why, 'infeasibility_reasons': why, 'I_x_eur': None,
                     'disposition': None, 'source': None, 'status': None, 'barrier_cause': None, 'Q_eur': None,
                     'F_eur': None, 'bar_eur': None, 'incumbent_bar_eur': inc['bar'], 'bar_sum_eur': None,
                     'sigma_Q_eur': sigma_q, 'resolution_eur': None, 'F_inc_minus_F_eur': None, 'outcome': None}
            if why:
                entry.update({'disposition': 'rejected_infeasible', 'F_eur': None,
                              'outcome': 'barrier_infeasible_not_evaluated'})
                cands.append(entry)
                continue
            z = lattice.canonical_z(raw)
            entry.update({'z': list(z), 'label': lattice.label(z), 'eval_key': key_of(z),
                          'I_x_eur': lattice.investment_cost(z),
                          'canonical': {'investment_year': lattice.year_of(z),
                                        'nodes': {str(n): list(v) for n, v in lattice.nodes_map(z).items()}}})
            if z == tuple(inc['z']):
                entry.update({'disposition': 'dropped_inactive_only', 'outcome': 'dropped'})
            elif entry['eval_key'] in seen:
                entry.update({'disposition': f"duplicate_of_direction_{seen[entry['eval_key']]}", 'outcome': 'dropped'})
            elif entry['eval_key'] in cache:
                seen[entry['eval_key']] = j
                entry['disposition'] = 'cache_hit'
            else:
                seen[entry['eval_key']] = j
                entry['disposition'] = 'new_evaluation'
            cands.append(entry)
        new = [c for c in cands if c['disposition'] == 'new_evaluation']
        record = {'poll_index': k, 'halton_t': t, 'halton_u': u, 'poll_size_delta': delta, 'mesh_size': MESH_SIZE,
                  'incumbent': inc_view, 'directions': [list(d) for d in dirs], 'candidates': cands,
                  'n_new_evaluations': len(new), 'n_cache_hits': sum(c['disposition'] == 'cache_hit' for c in cands),
                  'batches': [], 'decision': None, 'next_incumbent': None, 'next_poll_size': None,
                  'unit_poll': delta == DELTA_MIN}
        if n_new + len(new) > max_new_evaluations:
            record['decision'] = 'not_launched_evaluation_budget'
            _finish(record)
            termination = {'reason': 'evaluation_budget_exhausted', 'poll_size_reached': delta,
                           'detail': f'poll {k} needs {len(new)} new evaluations; {n_new} used of {max_new_evaluations}'}
            break
        stop = None
        for b in range(0, len(new), batch_size):
            batch = new[b:b + batch_size]
            recs = evaluate_fn([tuple(c['z']) for c in batch])
            if len(recs) != len(batch):
                raise RuntimeError('evaluate_fn returned a different number of records')
            for c, rec in zip(batch, recs):
                if rec.get('eval_key') != c['eval_key']:
                    raise RuntimeError(f"evaluation record eval key {rec.get('eval_key')} != {c['eval_key']}")
                cache[c['eval_key']] = dict(rec)
                n_new += 1
                if rec['status'] != 'certified':
                    n_barrier_new += 1
            record['batches'].append([c['label'] for c in batch])
            barrier_this_poll = sum(1 for c in new if c['eval_key'] in cache
                                    and cache[c['eval_key']]['status'] != 'certified')
            if barrier_this_poll >= BARRIER_STOP_PER_POLL or n_barrier_new >= BARRIER_STOP_OVERALL:
                stop = {'reason': 'STOP_FOR_REVIEW_barrier_rule', 'barrier_new_this_poll': barrier_this_poll,
                        'barrier_new_overall': n_barrier_new, 'rule': STOP_RULE}
                break
        for c in cands:
            if c['disposition'] not in ('cache_hit', 'new_evaluation'):
                continue
            rec = cache.get(c['eval_key'])
            if rec is None:  # not evaluated because the stop rule fired
                c['outcome'] = 'not_evaluated_stop_rule'
                continue
            c['source'] = rec.get('source')
            c['status'] = rec['status']
            c['barrier_cause'] = rec.get('barrier_cause')
            if rec['status'] != 'certified' or rec.get('Q') is None:
                c.update({'F_eur': None, 'outcome': 'barrier'})
                continue
            f_x = c['I_x_eur'] + rec['Q']
            res = resolution(rec.get('bar'), inc['bar'], sigma_q)
            c.update({'Q_eur': rec['Q'], 'F_eur': f_x, 'bar_eur': rec.get('bar'),
                      'bar_sum_eur': (rec['bar'] + inc['bar']) if (rec.get('bar') is not None
                                                                     and inc['bar'] is not None) else None,
                      'resolution_eur': res if math.isfinite(res) else None, 'F_inc_minus_F_eur': inc['F'] - f_x,
                      'outcome': classify(inc['F'], f_x, res)})
        if stop:
            record['decision'] = 'stopped_for_review'
            _finish(record)
            termination = dict(stop, poll_size_reached=delta)
            break
        improvers = [c for c in cands if c['outcome'] == 'improvement']
        if improvers:
            best = min(improvers, key=lambda c: (c['F_eur'], c['I_x_eur'], c['label']))
            inc = {'eval_key': best['eval_key'], 'z': tuple(best['z']), 'label': best['label'], 'I': best['I_x_eur'],
                   'Q': best['Q_eur'], 'F': best['F_eur'], 'bar': best['bar_eur'], 'source': best['source']}
            record.update({'decision': 'success', 'next_incumbent': best['label'], 'next_poll_size': delta * 2})
            delta *= 2
        elif delta == DELTA_MIN:
            record.update({'decision': 'failure_at_unit_poll_size', 'next_incumbent': inc['label'],
                           'next_poll_size': None})
            _finish(record)
            termination = {'reason': 'mesh_local_optimum_unit_poll_failed', 'poll_size_reached': delta}
            break
        else:
            record.update({'decision': 'failure', 'next_incumbent': inc['label'],
                           'next_poll_size': max(DELTA_MIN, delta // 2)})
            delta = max(DELTA_MIN, delta // 2)
        _finish(record)
        log(f"[PHASE-B] poll {k} Delta={record['poll_size_delta']} incumbent={record['incumbent']['label']} "
            f"new={record['n_new_evaluations']} hits={record['n_cache_hits']} -> {record['decision']}")
    if termination is None:
        termination = {'reason': 'max_polls_reached', 'poll_size_reached': delta}
    last = history[-1] if history else None
    unresolved = [c for c in (last['candidates'] if last else []) if c['outcome'] == 'indeterminate']
    return {'termination': termination,
            'incumbent': {kk: inc[kk] for kk in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar')},
            'n_polls': len(history), 'n_new_evaluations': n_new, 'n_barrier_new_evaluations': n_barrier_new,
            'final_poll_unresolved_indeterminate': [{k: c[k] for k in ('label', 'F_eur', 'F_inc_minus_F_eur',
                                                                        'resolution_eur')} for c in unresolved],
            'final_poll_feasible_points': [c['label'] for c in (last['candidates'] if last else []) if c['feasible']],
            'lattice_neighbourhood_of_incumbent': [
                {'label': lattice.label(z), 'eval_key': key_of(z), 'in_cache': key_of(z) in cache,
                 'polled_in_final_poll': any(c['eval_key'] == key_of(z) for c in (last['candidates'] if last else [])),
                 'I_x_eur': lattice.investment_cost(z)} for z in lattice.neighbourhood(tuple(inc['z']))],
            'history': history}


# ======================================================================================================================
#  inputs: unit costs, sigma_Q, the pinned x = 0 record
# ======================================================================================================================
def unit_costs_from_w2(payload):
    table = payload['expected_unit_costs_per_case_year']['new']
    return {int(y): {'power': v['power_eur_per_mva_discounted'], 'energy': v['energy_eur_per_mwh_discounted']}
            for y, v in table.items()}


def w2_cross_check(lattice, payload):
    """I(x) closed form vs every W2 candidate's I_new_eur (the production master's value)."""
    diffs = []
    for label, c in payload['candidates'].items():
        can = c['candidate_canonical']
        y = int(can['investment_year'])
        i_x = sum(lattice.c_p[y] * v[0] + lattice.c_e[y] * v[1] for v in can['nodes'].values())
        diffs.append((abs(i_x - c['I_new_eur']), label))
    worst = max(diffs)
    return {'n': len(diffs), 'max_abs_diff_eur': worst[0], 'worst_label': worst[1], 'ok': worst[0] <= 1e-6}


def sigma_q_from_tables(payload):
    t3 = payload['T3']
    return {'sigma_Q_eur': float(t3['residual_max_abs_eur']), 'residual_rms_eur': float(t3['residual_rms_eur']),
            'n_fit_points': t3.get('n'), 'source': dict(PHASE_A_TABLES),
            'note': ('measured under the C3 calibration (Phase A A3 fit, Addendum 28 "sigma_Q ~ 10-18k"); the '
                     'oracle configuration is unchanged; not re-measured under the baseline')}


def degradation_clause(lattice, sigma_q):
    """STEP4 5.2: does sigma_Q exceed one lattice step's investment cost? Unit-step costs per coordinate/year."""
    steps = {}
    for y in lattice.years:
        steps[f'P_{y}'] = lattice.c_p[y] * GRANULE_P_MVA
        steps[f'E_{y}'] = lattice.c_e[y] * GRANULE_E_MWH
    min_step = min(steps.values())
    return {'unit_step_costs_eur': steps, 'min_unit_step_cost_eur': min_step, 'sigma_Q_eur': sigma_q,
            'sigma_Q_exceeds_one_step': sigma_q > min_step, 'two_sigma_Q_below_min_step': 2 * sigma_q < min_step,
            'triggered': sigma_q > min_step,
            'note': ('the year coordinate has no fixed step cost (it scales with the capacity moved); its smallest '
                     'unit move on the domain is recorded separately as min_year_step_cost_eur')}


def min_year_step_cost(lattice):
    best = None
    for z in lattice.domain():
        if not lattice.has_storage(z) or z[-1] + 1 >= len(lattice.years):
            continue
        z2 = z[:-1] + (z[-1] + 1,)
        diff = abs(lattice.investment_cost(z) - lattice.investment_cost(z2))
        if best is None or diff < best[0]:
            best = (diff, lattice.label(z), lattice.label(z2))
    return {'min_year_step_cost_eur': best[0], 'between': [best[1], best[2]]} if best else None


def eval_key_of(canonical):
    return H.evaluation_key(H.candidate_key(canonical), {}, case_file_aa=CASE_FILE_AA,
                            ess_ageing_baseline=ESS_AGEING_BASELINE)


def canonical_of(lattice, z):
    return H.canonical_candidate(lattice.nodes_map(z), investment_year=lattice.year_of(z))


def make_key_of(lattice):
    memo = {}

    def key_of(z):
        z = lattice.canonical_z(z)
        if z not in memo:
            memo[z] = eval_key_of(canonical_of(lattice, z))
        return memo[z]
    return key_of


def x0_entry(lattice, a0_payload):
    a0 = a0_payload['points'][A0_RESULTS['label']]
    can = canonical_of(lattice, lattice.x0())
    if a0.get('status') != 'certified' or a0.get('candidate_key') != H.candidate_key(can):
        raise AssertionError(f"A0 x0 record not certified or not x = 0: {a0.get('status')} {a0.get('candidate_key')}")
    if a0.get('eval_key') == eval_key_of(can):
        raise AssertionError('the A0 (C3-era) x0 eval key equals the baseline key: the baseline is not in the key')
    return eval_key_of(can), {'label': 'x0', 'status': 'certified', 'Q': a0['certified_cost_gross_settlement_excluded'],
                              'bar': (a0.get('bar') or {}).get('value'), 'canonical': can, 'barrier_cause': None,
                              'source': {'kind': 'pinned_A0_x0_ageing_independent', 'path': A0_RESULTS['path'],
                                         'sha256': A0_RESULTS['sha256'], 'a0_eval_key_C3': a0.get('eval_key'),
                                         'certification_cycle': a0.get('certification_cycle'),
                                         'eval_dir': a0.get('eval_dir')}}


# ======================================================================================================================
#  the cache: committed BASELINE campaign results (and the C3 exclusion)
# ======================================================================================================================
def _git_state(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return tracked, not dirty


def spec_matches_baseline(spec, case_file_sha256, ess_params_sha256):
    cfg = spec.get('configuration') or {}
    checks = {
        'ess_ageing_baseline': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'case_file_aa': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_params_sha256': (cfg.get('ess_params_file') or {}).get('sha256') == ess_params_sha256,
        'case_file_sha256': cfg.get('case_file_sha256') == case_file_sha256,
        'overrides_empty': cfg.get('overrides') == {},
        'arm_label': cfg.get('arm_label') == ARM_LABEL,
        'cap': spec.get('cap') == CAP,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'no_model_variant': 'model_variant_label' not in spec and not any(
            'model_variant' in e for e in spec.get('candidates') or []),
        'no_per_entry_overrides': all((e.get('overrides') or {}) == {} for e in spec.get('candidates') or []),
    }
    return all(checks.values()), checks


def load_cache_source(results_rel, case_file_sha256, ess_params_sha256, repo=REPO, require_git=True):
    """One campaign_results.json -> (accepted, info, entries{eval_key: entry}). Pure apart from reading files."""
    path = os.path.join(repo, results_rel)
    info = {'path': results_rel, 'sha256': H.sha256_file(path)}
    if require_git:
        tracked, clean = _git_state(results_rel)
        info.update({'git_tracked': tracked, 'git_clean': clean})
        if not (tracked and clean):
            return False, dict(info, reason='not git-tracked and clean'), {}
    with open(path) as handle:
        results = json.load(handle)
    spec_rel = results.get('campaign_spec_path')
    spec_path = os.path.join(repo, spec_rel) if spec_rel else None
    if not spec_path or not os.path.isfile(spec_path):
        return False, dict(info, reason=f'campaign spec missing: {spec_rel}'), {}
    if H.sha256_file(spec_path) != results.get('campaign_spec_sha256'):
        return False, dict(info, reason='campaign spec sha256 differs from the results file'), {}
    with open(spec_path) as handle:
        spec = json.load(handle)
    ok, checks = spec_matches_baseline(spec, case_file_sha256, ess_params_sha256)
    info.update({'campaign_id': spec.get('campaign_id'), 'campaign_spec_path': spec_rel,
                 'campaign_spec_sha256': results.get('campaign_spec_sha256'), 'spec_checks': checks})
    if require_git and spec_rel:
        tracked, clean = _git_state(spec_rel)
        info.update({'spec_git_tracked': tracked, 'spec_git_clean': clean})
        ok = ok and tracked and clean
    manifest_rel = os.path.join(os.path.dirname(results_rel), 'campaign_manifest_sha256.json')
    if os.path.isfile(os.path.join(repo, manifest_rel)):
        with open(os.path.join(repo, manifest_rel)) as handle:
            manifest = json.load(handle)
        records = {k: v for k, v in manifest.items()
                   if os.path.basename(k) in ('evaluation_record.json', 'per_cycle_record.jsonl')}
        bad = sorted(k for k, v in records.items() if not os.path.isfile(os.path.join(repo, k))
                     or H.sha256_file(os.path.join(repo, k)) != v)
        info['manifest'] = {'path': manifest_rel, 'sha256': H.sha256_file(os.path.join(repo, manifest_rel)),
                            'n_record_files_checked': len(records), 'mismatched': bad}
        if bad:
            ok = False
    elif require_git:
        ok = False
        info['manifest'] = {'path': manifest_rel, 'present': False}
    if not ok:
        return False, dict(info, reason='spec does not declare the SAME baseline configuration'), {}
    entries, skipped = {}, []
    for label, p in sorted((results.get('points') or {}).items()):
        status = p.get('status')
        if status in ('not_launched_stop_rule', None):
            skipped.append({'label': label, 'status': status})
            continue
        can = p.get('candidate_canonical')
        if not can:
            raise AssertionError(f'{results_rel}:{label}: status {status} without a candidate_canonical')
        ekey = eval_key_of(can)
        if p.get('eval_key') != ekey:
            raise AssertionError(f'{results_rel}:{label}: eval key {p.get("eval_key")} != recomputed {ekey}')
        certified = status == 'certified'
        q = p.get('certified_cost_gross_settlement_excluded') if certified else None
        if certified and q is None:
            raise AssertionError(f'{results_rel}:{label}: certified without a Q')
        entry = {'label': label, 'status': status, 'Q': q, 'bar': (p.get('bar') or {}).get('value'),
                 'canonical': can, 'barrier_cause': p.get('barrier_cause'),
                 'source': {'kind': 'baseline_campaign', 'path': results_rel, 'label': label,
                            'eval_dir': p.get('eval_dir'), 'campaign_id': spec.get('campaign_id')}}
        if ekey in entries:
            raise AssertionError(f'{results_rel}: eval key {ekey[:16]} twice in one file')
        entries[ekey] = entry
    info.update({'n_entries': len(entries), 'skipped_not_evaluations': skipped})
    return True, info, entries


def merge_cache(sources):
    """sources: list of entries dicts. Duplicate eval keys must agree bitwise on status, Q and bar."""
    cache, duplicates = {}, []
    for entries in sources:
        for ekey, e in entries.items():
            if ekey in cache:
                a = cache[ekey]
                same = (a['status'], a['Q'], a['bar']) == (e['status'], e['Q'], e['bar'])
                duplicates.append({'eval_key': ekey, 'labels': [a['label'], e['label']],
                                   'sources': [a['source'].get('path'), e['source'].get('path')],
                                   'status_Q_bar_bitwise_identical': same,
                                   'Q': [a['Q'], e['Q']], 'bar': [a['bar'], e['bar']]})
                if not same:
                    raise AssertionError(f'duplicate eval key {ekey[:16]} with different status/Q/bar: '
                                         f'{duplicates[-1]} -- a determinism failure; STOP for review')
                cache[ekey].setdefault('duplicate_sources', []).append(e['source'])
                continue
            cache[ekey] = dict(e)
    return cache, duplicates


def c3_exclusion(cache_keys, domain_keys, repo=REPO, globs=C3_ERA_GLOBS):
    """C3-era results are NOT cache: none of their eval keys may equal a cache / domain key, none of their specs
    may declare the baseline. Scope: every campaign_results.json under the globs (recorded)."""
    files, collisions, declared = [], [], []
    for pattern in globs:
        for path in sorted(glob.glob(os.path.join(repo, pattern), recursive=True)):
            rel = os.path.relpath(path, repo)
            with open(path) as handle:
                payload = json.load(handle)
            points = payload.get('points') or {}
            points = points.values() if isinstance(points, dict) else points
            keys = {p.get('eval_key') or p.get('candidate_key') for p in points if isinstance(p, dict)}
            keys.discard(None)
            spec_rel = payload.get('campaign_spec_path')
            spec_declares = None
            if spec_rel and os.path.isfile(os.path.join(repo, spec_rel)):
                with open(os.path.join(repo, spec_rel)) as handle:
                    spec_declares = (json.load(handle).get('configuration') or {}).get('ess_ageing_baseline') is not None
                if spec_declares:
                    declared.append(rel)
            hit = sorted(k for k in keys if k in cache_keys or k in domain_keys)
            if hit:
                collisions.append({'file': rel, 'keys': hit})
            files.append({'file': rel, 'n_keys': len(keys), 'spec_declares_ess_ageing_baseline': spec_declares})
    return {'scope_globs': list(globs), 'files': files, 'n_files': len(files), 'key_collisions': collisions,
            'specs_declaring_a_baseline': declared, 'ok': not collisions and not declared and bool(files)}


# ======================================================================================================================
#  pinned inputs, preconditions, memory (the S47 launcher's measures, transcribed -- not imported: its module-level
#  parent guard would stack on this one)
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v17', SPEC_V17), ('spec_v15', SPEC_V15), ('ess_params_file', ESS_PARAMS_FILE),
                      ('cost_file', COST_FILE), ('a0_results', A0_RESULTS),
                      ('investment_cost_results', INVESTMENT_COST_RESULTS), ('phase_a_tables', PHASE_A_TABLES)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked, clean = _git_state(pin['path'])
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': clean}
        if not (got == pin['sha256'] and tracked and clean):
            failures.append(f'{name}: {out[name]}')
    for name, pin in (('identity_checks', IDENTITY_CHECKS), ('case_file_gate', CASE_FILE_GATE),
                      ('x0_diff_analysis', X0_ANALYSIS), ('baseline_launcher', BASELINE_LAUNCHER)):
        path = os.path.join(REPO, pin['path'])
        present = os.path.isfile(path)
        tracked, clean = _git_state(pin['path']) if present else (False, False)
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        entry = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': in_head, 'git_tracked': tracked,
                 'git_clean': clean, 'sha256': H.sha256_file(path) if present else None}
        ok = present and tracked and clean and in_head
        if present and name != 'baseline_launcher':
            payload = _load(pin['path'])
            failing = sorted(k for k, v in (payload.get('checks') or {}).items() if not v)
            entry.update({'all_ok': payload.get('all_ok'), 'failing_checks': failing, 'expect': pin['expect']})
            if name == 'case_file_gate':
                ok = ok and payload.get('all_ok') is False and failing == GATE_EXPECTED_FAILING
            else:
                ok = ok and payload.get('all_ok') is True
        out[name] = entry
        if not ok:
            failures.append(f'{name}: {entry}')
    return out, failures


def _check_baseline_launcher_declaration():
    """The declaration here must be the S2/S3 launcher's, byte for byte (read as text, never imported)."""
    with open(os.path.join(REPO, BASELINE_LAUNCHER['path'])) as handle:
        src = handle.read()
    ns = {}
    for node in ast.parse(src).body:  # top-level literal assignments only
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('CASE_FILE_AA', 'ESS_AGEING_BASELINE'):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    if set(ns) != {'CASE_FILE_AA', 'ESS_AGEING_BASELINE'}:
        return False, f'declarations not found in {BASELINE_LAUNCHER["path"]}: {sorted(ns)}'
    ok = ns['CASE_FILE_AA'] == CASE_FILE_AA and ns['ESS_AGEING_BASELINE'] == ESS_AGEING_BASELINE
    return ok, {'launcher': ns, 'equal': ok}


def _check_case_file_loads_to_declaration():
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded = params.admm.anderson_acceleration
    return {'loaded': loaded, 'equals_declaration': loaded == CASE_FILE_AA}


GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 11 * GIB // 4
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE = (f'hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {CONCURRENCY} x '
               f'{MEMORY_PER_CHILD_BUDGET_BYTES / GIB:g} GiB (the S47 launcher rule)')


def memory_preflight():
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
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
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    if any(raw.get(k) is None for k in need):
        return {'pass': False, 'missing': [k for k in need if raw.get(k) is None], 'rule': MEMORY_RULE}
    avail = total - sum(raw[k] for k in need) * page
    return {'utc': datetime.now(timezone.utc).isoformat(), 'available_gib': avail / GIB,
            'required_gib': MEMORY_REQUIRED_BYTES / GIB, 'pass': avail >= MEMORY_REQUIRED_BYTES, 'rule': MEMORY_RULE,
            'pages': {k: raw.get(k) for k in need + ('Pages free', 'Pages inactive')}, 'page_size': page}


# ======================================================================================================================
#  rule eleven
# ======================================================================================================================
def rule_eleven(lattice, sigma_q):
    """Capture paths for every quantity the Phase B record needs, asserted BEFORE running."""
    harness_checks = H.assert_record_capture_paths()
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    checks = {
        'record_status': "'status': status" in build_src,
        'record_barrier_cause': "'barrier_cause': cause" in build_src,
        'record_Q_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'record_bar': "'bar': bar" in build_src and 'gross_step_abs' in inspect.getsource(H._max_step_last_n),
        'record_cycles_and_certification_cycle': ("'cycles_run': len(rows)" in build_src
                                                  and "'certification_cycle':" in build_src),
        'record_rule_ten': "'terminal_step_over_threshold':" in build_src,
        'record_wall_and_rss': "'wall_time_s': wall" in build_src and "'peak_rss': peak_rss" in build_src,
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'ess_ageing_readback_terminal': "holder['ess_ageing_readback_terminal'] = ess_ageing_readback_models(" in child_src,
        'parent_barrier_record_for_missing': "'status': 'harness_error', 'barrier': True" in inspect.getsource(
            H._barrier_record_for_missing),
    }
    # Phase B's own record: build one synthetic poll with the REAL loop and assert every field is produced.
    key_of = make_key_of(lattice)
    z_inc = lattice.x0()
    fake_cache = {key_of(z_inc): {'label': 'x0', 'status': 'certified', 'Q': 1.0e8, 'bar': 1.0, 'source': 'synthetic',
                                  'canonical': canonical_of(lattice, z_inc)}}
    inc = {'eval_key': key_of(z_inc), 'z': z_inc, 'label': 'x0', 'I': 0.0, 'Q': 1.0e8, 'F': 1.0e8, 'bar': 1.0,
           'source': 'synthetic'}

    def _no_eval(batch):  # synthetic barrier records: exercises the record path without any solve
        return [{'label': lattice.label(z), 'status': 'not_certified', 'eval_key': key_of(z), 'Q': None, 'bar': None,
                 'canonical': canonical_of(lattice, z), 'barrier_cause': 'synthetic', 'source': 'synthetic'}
                for z in batch]
    try:
        out = run_mads(lattice, dict(fake_cache), key_of, inc, _no_eval, sigma_q, max_polls=1, log=lambda m: None)
        poll = out['history'][0]
        checks['poll_record_fields'] = all(f in poll for f in POLL_RECORD_FIELDS)
        checks['candidate_record_fields'] = all(all(f in c for f in CANDIDATE_RECORD_FIELDS) for c in poll['candidates'])
        checks['terminal_record_fields'] = all(f in out for f in ('termination', 'incumbent', 'history',
                                                                    'final_poll_unresolved_indeterminate',
                                                                    'lattice_neighbourhood_of_incumbent'))
    except AssertionError:
        checks['poll_record_fields'] = checks['candidate_record_fields'] = checks['terminal_record_fields'] = False
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (Phase B record): capture paths missing: {missing}')
    return {'checks': checks, 'harness_record_capture_checklist': harness_checks,
            'poll_record_fields': list(POLL_RECORD_FIELDS), 'candidate_record_fields': list(CANDIDATE_RECORD_FIELDS)}


# ======================================================================================================================
#  inputs shared by --freeze and --run
# ======================================================================================================================
def build_inputs(own_root_rel):
    failures, ev = [], {}
    pins, more = _check_pins()
    ev['pins'] = pins
    failures += more
    ok, decl = _check_baseline_launcher_declaration()
    ev['declaration_equals_s47_launcher'] = decl
    if not ok:
        failures.append(f'declaration differs from {BASELINE_LAUNCHER["path"]}: {decl}')
    case = _check_case_file_loads_to_declaration()
    ev['case_file_aa'] = case
    if not case['equals_declaration']:
        failures.append(f"case file AA {case['loaded']} != {CASE_FILE_AA}")
    loaded = H.load_ess_ageing_parameters(os.path.join(REPO, H.ESS_PARAMS_FILE_REL))
    if H.ess_ageing_canonical_text(loaded) != H.ess_ageing_canonical_text(ESS_AGEING_BASELINE):
        failures.append(f'ESS params file does not load to the declaration: {loaded}')
    v17 = _load(SPEC_V17['path'])
    ev['spec_v17_S4'] = [s for s in v17['steps'] if s['id'] == 'S4']
    ev['spec_v17_S4_prediction'] = v17['predictions_recorded_before_run'].get('S4')
    ev['spec_v15_master_constraints'] = _load(SPEC_V15['path']).get('master_constraints')
    w2 = _load(INVESTMENT_COST_RESULTS['path'])
    costs = unit_costs_from_w2(w2)
    years = tuple(sorted(costs))
    if years != (2025, 2030, 2035):
        failures.append(f'instance years from the W2 table {years} != (2025, 2030, 2035)')
    if (w2.get('master_facts') or {}).get('budget_eur') != BUDGET_EUR or \
            (w2.get('master_facts') or {}).get('max_capacity_mwh') != E_MAX_MWH:
        failures.append(f"W2 master facts differ: {w2.get('master_facts')}")
    lattice = Lattice(years, costs)
    ev['I_x_cross_check_vs_W2'] = w2_cross_check(lattice, w2)
    if not ev['I_x_cross_check_vs_W2']['ok']:
        failures.append(f"I(x) closed form != W2 table: {ev['I_x_cross_check_vs_W2']}")
    sq = sigma_q_from_tables(_load(PHASE_A_TABLES['path']))
    ev['sigma_Q'] = sq
    ev['degradation_clause'] = dict(degradation_clause(lattice, sq['sigma_Q_eur']), **(min_year_step_cost(lattice) or {}))
    if ev['degradation_clause']['triggered']:
        failures.append(f"STEP4 5.2 degradation clause triggers: {ev['degradation_clause']} -- not implemented; STOP")
    key_of = make_key_of(lattice)
    domain = lattice.domain()
    ev['domain'] = {'n_points': len(domain), 'n_storage_points': len(domain) - 1,
                    'per_year': {y: sum(1 for z in domain if lattice.has_storage(z) and lattice.years[z[-1]] == y)
                                 for y in years}}
    # x = 0, pinned
    x0_key, x0 = x0_entry(lattice, _load(A0_RESULTS['path']))
    # the cache
    case_sha = H.sha256_file(H.CASE_FILE)
    sources, accepted, rejected = [], [], []
    for path in sorted(glob.glob(os.path.join(REPO, CACHE_SOURCE_GLOB))):
        rel = os.path.relpath(path, REPO)
        if os.path.dirname(rel) == own_root_rel:
            continue
        ok_src, info, entries = load_cache_source(rel, case_sha, ESS_PARAMS_FILE['sha256'])
        (accepted if ok_src else rejected).append(info)
        if ok_src:
            sources.append(entries)
    if x0_key in {k for s in sources for k in s}:
        failures.append('a baseline campaign holds x = 0 -- x = 0 is not re-run by design; STOP for review')
    cache, duplicates = merge_cache(sources + [{x0_key: x0}])
    ev['cache_sources_accepted'] = accepted
    ev['cache_sources_rejected'] = rejected
    ev['cache_duplicates'] = duplicates
    if not accepted:
        failures.append('no baseline campaign results accepted as cache')
    ev['c3_exclusion'] = c3_exclusion(set(cache) - {x0_key}, {key_of(z) for z in domain} - {x0_key})
    if not ev['c3_exclusion']['ok']:
        failures.append(f"C3-era exclusion failed: {ev['c3_exclusion']['key_collisions']} "
                        f"{ev['c3_exclusion']['specs_declaring_a_baseline']}")
    inc, inc_record = initial_incumbent(lattice, cache, x0_key, sq['sigma_Q_eur'])
    ev['initial_incumbent'] = inc_record
    ev['cache_table'] = {k: {kk: v.get(kk) for kk in ('label', 'status', 'Q', 'bar', 'canonical', 'source')}
                         for k, v in sorted(cache.items())}
    try:
        ev['rule_eleven'] = rule_eleven(lattice, sq['sigma_Q_eur'])
    except AssertionError as error:
        failures.append(str(error))
    return failures, ev, lattice, key_of, domain, cache, inc, x0_key


def expected_first_poll(lattice, cache, key_of, inc, sigma_q):
    """Zero-solve dry run of the poll sequence from the incumbent: follows the loop as long as no new evaluation
    is needed (cache hits and infeasible points are known); stops at the first poll that needs one."""
    needed = []

    def _dry_eval(batch):
        needed.extend(batch)
        raise _DryStop()
    try:
        out = run_mads(lattice, dict(cache), key_of, inc, _dry_eval, sigma_q, log=lambda m: None)
        return {'complete_without_new_evaluations': True, 'result': out}
    except _DryStop:
        return {'complete_without_new_evaluations': False, 'first_new_evaluations': [lattice.label(z) for z in needed]}


class _DryStop(Exception):
    pass


def _summarize_poll(poll):
    return {'poll_index': poll['poll_index'], 'Delta': poll['poll_size_delta'], 'halton_t': poll['halton_t'],
            'incumbent': poll['incumbent']['label'], 'decision': poll['decision'],
            'candidates': [{'d': c['direction'], 'label': c['label'], 'disposition': c['disposition'],
                            'reasons': c['infeasibility_reasons'], 'outcome': c['outcome']} for c in poll['candidates']]}


# ======================================================================================================================
#  --freeze / --run
# ======================================================================================================================
def campaign_root(campaign_id):
    return os.path.join(REPO, _P47, f'campaign_{campaign_id}')


def _extra(ev, domain, lattice, key_of, dry):
    return {'campaign_script': os.path.basename(__file__), 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
            'label': LABEL, 'stage': STAGE, 'spec_v17': dict(SPEC_V17), 'spec_v17_S4': ev['spec_v17_S4'],
            'spec_v17_S4_prediction': ev['spec_v17_S4_prediction'], 'pins': ev['pins'],
            'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE, 'stop_rule': STOP_RULE,
            'master_problem': {'nodes': list(ACTIVE_NODES), 'years': list(lattice.years), 'granule_P_mva': GRANULE_P_MVA,
                               'granule_E_mwh': GRANULE_E_MWH, 'E_max_mwh': E_MAX_MWH, 'duration_h': [2, 4],
                               'budget_eur': BUDGET_EUR, 'form': 'single-cohort, ONE common investment year (n = 7)',
                               'unit_costs_discounted': {str(y): {'power': lattice.c_p[y], 'energy': lattice.c_e[y]}
                                                         for y in lattice.years},
                               'spec_v15_master_constraints': ev['spec_v15_master_constraints']},
            'poll_design': {'design': POLL_DESIGN, 'n_vars': N_VARS, 'n_directions': N_VARS + 1, 'primes': list(PRIMES[:N_VARS]),
                            'halton_t0': HALTON_T0, 'halton_index': 't = t0 + k (k = poll counter from 0)',
                            'rounding': 'd = round_half_away(Delta h / ||h||_inf)', 'delta_0': DELTA_0,
                            'delta_min': DELTA_MIN, 'mesh_size': MESH_SIZE, 'success': 'Delta *= 2',
                            'failure': 'Delta = max(1, Delta // 2)', 'full_poll': True, 'batch_size': CONCURRENCY,
                            'max_new_evaluations': MAX_NEW_EVALUATIONS, 'max_polls': MAX_POLLS},
            'sigma_Q': ev['sigma_Q'], 'degradation_clause': ev['degradation_clause'],
            'I_x_cross_check_vs_W2': ev['I_x_cross_check_vs_W2'], 'domain_summary': ev['domain'],
            'domain_I_x_eur': {lattice.label(z): lattice.investment_cost(z) for z in domain},
            'cache_sources_accepted': ev['cache_sources_accepted'], 'cache_sources_rejected': ev['cache_sources_rejected'],
            'cache_duplicates': ev['cache_duplicates'], 'cache_table': ev['cache_table'],
            'c3_exclusion': ev['c3_exclusion'], 'initial_incumbent': ev['initial_incumbent'],
            'declaration_equals_s47_launcher': ev['declaration_equals_s47_launcher'],
            'expected_first_poll_dry_run': dry, 'rule_eleven': ev['rule_eleven']['checks'],
            'memory_preflight_rule': MEMORY_RULE}


def _dry_summary(lattice, cache, key_of, inc, sigma_q):
    dry = expected_first_poll(lattice, cache, key_of, inc, sigma_q)
    if dry['complete_without_new_evaluations']:
        res = dry['result']
        return {'complete_without_new_evaluations': True, 'termination': res['termination'],
                'incumbent': res['incumbent']['label'], 'polls': [_summarize_poll(p) for p in res['history']],
                'lattice_neighbourhood_of_incumbent': res['lattice_neighbourhood_of_incumbent']}
    return dry


def freeze(campaign_id, started):
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, ev, lattice, key_of, domain, cache, inc, _x0 = build_inputs(own_rel)
    failures += more
    if failures:
        for f in failures:
            _log(f'[S47-PHASE-B FREEZE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    dry = _dry_summary(lattice, cache, key_of, inc, ev['sigma_Q']['sigma_Q_eur'])
    entries = [(lattice.label(z), lattice.nodes_map(z), {'investment_year': lattice.year_of(z)}) for z in domain]
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, campaign_id, entries,
        configuration={'name': f'{LABEL}: the case file alone (AA keep_memory) with the ESS ageing parameters declared',
                       'arm_label': ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('candidates = the WHOLE admissible domain (budget-, bound- and duration-feasible '
                                'common-year lattice points); only polled, non-cached points are evaluated')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=_extra(ev, domain, lattice, key_of, dry))
    with open(spec_path) as handle:
        spec = json.load(handle)  # validate what is ON DISK
    checks = validate_spec(spec, lattice, key_of, domain, ev)
    guard = PARENT_GUARD.verify(0)
    _log(f'[S47-PHASE-B] {LABEL}')
    _log(f'[S47-PHASE-B] frozen spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha} ({len(domain)} domain entries)')
    _log(f"[S47-PHASE-B] sigma_Q={ev['sigma_Q']['sigma_Q_eur']}; degradation clause {ev['degradation_clause']}")
    _log(f"[S47-PHASE-B] cache sources accepted: {[(a['path'], a['sha256'], a['n_entries']) for a in ev['cache_sources_accepted']]}")
    _log(f"[S47-PHASE-B] cache sources rejected: {[(r['path'], r.get('reason')) for r in ev['cache_sources_rejected']]}")
    _log(f"[S47-PHASE-B] duplicates: {ev['cache_duplicates']}")
    _log(f"[S47-PHASE-B] C3 exclusion: ok={ev['c3_exclusion']['ok']} files={ev['c3_exclusion']['n_files']}")
    _log(f"[S47-PHASE-B] initial incumbent: {ev['initial_incumbent']['incumbent']} "
         f"every storage F > F(0): {ev['initial_incumbent']['every_storage_point_F_gt_F0']}")
    _log(f'[S47-PHASE-B] expected polls (dry run, zero solves): {json.dumps(dry, default=str)}')
    _log(f"[S47-PHASE-B] spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}")
    _log(f'[S47-PHASE-B] parent guard {PARENT_GUARD.counts} verify0_failures={guard} wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard
    _log(f'[S47-PHASE-B] freeze {"OK" if ok else "NOT OK"}; run with: --campaign-id {campaign_id} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


def validate_spec(spec, lattice, key_of, domain, ev):
    cfg = spec['configuration']
    extra = spec.get('extra') or {}
    labels = [e['label'] for e in spec['candidates']]
    checks = {
        'n_entries_equal_domain': len(spec['candidates']) == len(domain),
        'entries_in_domain_order': labels == [lattice.label(z) for z in domain],
        'eval_keys_recompute': all(e['eval_key'] == key_of(z) for e, z in zip(spec['candidates'], domain)),
        'no_entry_is_the_undeclared_key': all(e['eval_key'] != H.evaluation_key(e['key'], {}, case_file_aa=CASE_FILE_AA)
                                              for e in spec['candidates']),
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_params_pinned': (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256'],
        'overrides_empty': cfg.get('overrides') == {} and all(e.get('overrides') == {} for e in spec['candidates']),
        'no_model_variant': 'model_variant_label' not in spec,
        'post_certification_none': all(e.get('post_certification') is None for e in spec['candidates']),
        'cap_500': spec.get('cap') == CAP, 'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'ten_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_s39_D': cfg.get('arm_label') == ARM_LABEL,
        'cache_table_frozen': extra.get('cache_table') == json.loads(json.dumps(ev['cache_table'])),
        'cache_sources_frozen': ([(a['path'], a['sha256']) for a in extra.get('cache_sources_accepted') or []]
                                 == [(a['path'], a['sha256']) for a in ev['cache_sources_accepted']]),
        'initial_incumbent_frozen': (extra.get('initial_incumbent') or {}).get('incumbent') == json.loads(
            json.dumps(ev['initial_incumbent']['incumbent'])),
        'sigma_Q_frozen': (extra.get('sigma_Q') or {}).get('sigma_Q_eur') == ev['sigma_Q']['sigma_Q_eur'],
        'poll_design_frozen': (extra.get('poll_design') or {}).get('halton_t0') == HALTON_T0
                              and (extra.get('poll_design') or {}).get('delta_0') == DELTA_0
                              and (extra.get('poll_design') or {}).get('max_new_evaluations') == MAX_NEW_EVALUATIONS,
        'domain_I_x_frozen': all(abs(extra.get('domain_I_x_eur', {}).get(lattice.label(z), -1.0)
                                     - lattice.investment_cost(z)) <= 1e-6 for z in domain),
        'c3_exclusion_ok': (extra.get('c3_exclusion') or {}).get('ok') is True,
    }
    return checks


def _record_from_harness(rec, ekey):
    rec = rec or {}
    certified = rec.get('status') == 'certified'
    verified = rec.get('ess_ageing_verified_pre_run') or {}
    pc_path = rec.get('per_cycle_record_path')
    return {'label': rec.get('candidate_label'), 'status': rec.get('status'), 'eval_key': rec.get('eval_key', ekey),
            'Q': rec.get('certified_cost') if certified else None, 'bar': (rec.get('bar') or {}).get('value'),
            'canonical': rec.get('candidate_canonical'), 'barrier_cause': rec.get('barrier_cause'),
            'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
            'rule_ten_terminal_step_over_threshold': (rec.get('rule_ten') or {}).get('terminal_step_over_threshold'),
            'ess_ageing_readback_pre_run_all_match': (verified.get('readback_pre_run') or {}).get('all_match'),
            'ess_ageing_readback_terminal_all_match': (rec.get('ess_ageing_readback_terminal') or {}).get('all_match'),
            'wall_time': rec.get('wall_time_s'), 'peak_rss': rec.get('peak_rss'), 'eval_dir': rec.get('eval_dir'),
            'per_cycle_record': {'path': pc_path, 'sha256': H.sha256_file(os.path.join(REPO, pc_path))
                                 if pc_path and os.path.isfile(os.path.join(REPO, pc_path)) else None},
            'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
            'source': {'kind': 'phase_b_new_evaluation', 'eval_dir': rec.get('eval_dir')}}


def run(campaign_id, started, spec_sha256):
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {sorted(os.listdir(root))}')
    more, ev, lattice, key_of, domain, cache, inc, x0_key = build_inputs(own_rel)
    failures += more
    checks = validate_spec(spec, lattice, key_of, domain, ev)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, frozen, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('this script', spec['extra'].get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__)))):
        if frozen != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    memory = memory_preflight()
    _log(f'[S47-PHASE-B] memory preflight {memory}')
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {memory}')
    if failures:
        for f in failures:
            _log(f'[S47-PHASE-B PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    label_of_key = {e['eval_key']: e['label'] for e in spec['candidates']}
    sigma_q = ev['sigma_Q']['sigma_Q_eur']
    state_path = os.path.join(root, 'phase_b_state.json')
    lock = H.acquire_campaign_lock(campaign_id, spec_sha256)
    _log(f'[S47-PHASE-B] campaign lock acquired: {lock}; git HEAD {head}')
    new_points, batch_infos, history_so_far = {}, [], []
    ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)

    def evaluate_fn(batch):
        keys = [key_of(z) for z in batch]
        labels = [label_of_key[k] for k in keys]
        _log(f'[S47-PHASE-B] evaluating {labels}')
        H.evaluate.last_batch_info = {}
        recs = H.evaluate(labels, ctx)
        batch_infos.append({'labels': labels, **getattr(H.evaluate, 'last_batch_info', {})})
        out = []
        for k, lab, rec in zip(keys, labels, recs):
            r = _record_from_harness(rec, k)
            new_points[lab] = r
            out.append(r)
        return out

    def on_poll(record):
        history_so_far.append(record)
        H._atomic_write_json(state_path, {'spec_sha256': spec_sha256, 'utc': datetime.now(timezone.utc).isoformat(),
                                          'history': history_so_far, 'new_points': new_points})

    try:
        result = run_mads(lattice, cache, key_of, inc, evaluate_fn, sigma_q, on_poll=on_poll)
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S47-PHASE-B] campaign lock released')
    harness_errors = [lab for lab, r in new_points.items() if r['status'] not in ('certified', 'not_certified')
                      or r['exit_code'] != 0]
    readback_mismatch = [lab for lab, r in new_points.items() if r['status'] in ('certified', 'not_certified') and not (
        r['ess_ageing_readback_pre_run_all_match'] and r['ess_ageing_readback_terminal_all_match'])]
    guard = PARENT_GUARD.verify(0)
    results = {
        'LABEL': LABEL, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE, 'stop_rule': STOP_RULE,
        'STOP_FOR_REVIEW': result['termination']['reason'].startswith('STOP_FOR_REVIEW'),
        'termination': result['termination'], 'final_incumbent': result['incumbent'],
        'initial_incumbent': ev['initial_incumbent'], 'sigma_Q': ev['sigma_Q'],
        'n_polls': result['n_polls'], 'n_new_evaluations': result['n_new_evaluations'],
        'n_barrier_new_evaluations': result['n_barrier_new_evaluations'],
        'final_poll_unresolved_indeterminate': result['final_poll_unresolved_indeterminate'],
        'final_poll_feasible_points': result['final_poll_feasible_points'],
        'lattice_neighbourhood_of_incumbent': result['lattice_neighbourhood_of_incumbent'],
        'claim_scope': ('mesh-local optimality is claimed only over the POLLED directions of the final unit poll '
                        '(STEP4 3, 8); lattice_neighbourhood_of_incumbent lists every feasible ||dz||_inf <= 1 '
                        'neighbour with whether it was polled or cached'),
        'poll_history': result['history'],
        # cache-compatible schema, so a later baseline campaign / resume reads these as cache (load_cache_source)
        'points': {lab: {'status': r['status'], 'eval_key': r['eval_key'], 'candidate_canonical': r['canonical'],
                         'certified_cost_gross_settlement_excluded': r['Q'], 'bar': {'value': r['bar']},
                         'barrier_cause': r['barrier_cause'], 'eval_dir': r['eval_dir'], 'detail': r}
                   for lab, r in new_points.items()},
        'harness_errors': harness_errors, 'readback_mismatch_points': readback_mismatch, 'batch_info': batch_infos,
        'pre_run_evidence': {k: ev[k] for k in ('pins', 'case_file_aa', 'cache_sources_accepted', 'c3_exclusion',
                                                'I_x_cross_check_vs_W2', 'degradation_clause')},
        'rule_eleven_asserted_before_run': ev['rule_eleven']['checks'],
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f"[S47-PHASE-B] termination {result['termination']}; incumbent {result['incumbent']['label']} "
         f"F={result['incumbent']['F']}; polls {result['n_polls']}; new evaluations {result['n_new_evaluations']}")
    if guard or harness_errors or readback_mismatch:
        _log(f'[S47-PHASE-B] NOT OK guard={guard} harness_errors={harness_errors} readback={readback_mismatch}')
        sys.exit(1)
    reason = result['termination']['reason']
    if reason.startswith('STOP_FOR_REVIEW'):
        sys.exit(3)
    if reason != 'mesh_local_optimum_unit_poll_failed':
        sys.exit(2)
    _log('[S47-PHASE-B] OK: terminated by the unit-poll failure')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='run the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--campaign-id', default=DEFAULT_CAMPAIGN_ID)
    args = parser.parse_args()
    if not CAMPAIGN_ID_PATTERN.match(args.campaign_id):
        parser.error(f'--campaign-id must match {CAMPAIGN_ID_PATTERN.pattern}')
    started = time.time()
    os.chdir(REPO)
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(args.campaign_id, started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(args.campaign_id, started, args.spec_sha256)


if __name__ == '__main__':
    main()
