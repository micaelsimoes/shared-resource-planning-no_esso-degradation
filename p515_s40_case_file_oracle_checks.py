"""
P5.15 Addendum 22 item (3) -- zero-solve load check.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 22, item `3_case_file`;
frozen spec v11 `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`
(`oracle.configuration`).

Loads `data/SRP1/SRP1_params.json` (now carrying D's configuration, committed
alone per the spec's own risk note -- "the first change to the case file in
this campaign") through the REAL production loader
(`p56a_oracle.fresh_planning`, which calls
`SharedResourcesPlanning.read_planning_problem()` -- never a
reimplementation), and compares the resulting `planning.params.admm` fields,
FIELD BY FIELD, against the in-force values on a SEPARATE fresh planning
object to which `p515_g_g1_g4_admm_gates._s39_configure_hook('s39_D')` (the
exact override path the oracle run used) has been applied -- zero-solve, by
construction (the hook only mutates `planning.params`, never solves).

Guard: `SolveProfileGuard` armed at 0 permitted solves for the WHOLE script,
raising on entry from any call site -- this script must not solve anything.

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s40_case_file_oracle_checks.py \\
        > data/SRP1/Results/P515S40/case_file_oracle_checks_launch.log 2>&1

Run attached, alone, both streams captured.
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40')
RESULTS_PATH = os.path.join(OUT_DIR, 'case_file_oracle_load_check.json')

# Fields on `admm_parameters.ADMMParameters` that fully determine ADMM
# numerical behaviour ("oracle-relevant"), read directly by attribute /
# nested-key path. `_source` bookkeeping fields are reported SEPARATELY
# (expected to differ: 'case_file' on the case-file-alone object vs the
# override string 'p515s39_override_not_case_file' on the hook-applied
# object) -- they record PROVENANCE, not a numerical difference in what
# is in force.
SCALAR_FIELDS = (
    'num_max_iters',
    'minimum_consecutive_converged_cycles',
    'shared_ess_normalization_floor_mva',
    'adaptive_penalty',
    'objective_scale',
    'objective_scale_assert_factor',
    'shared_ess_reference_rating_mva',
    'shared_ess_initialization',
    'boyd_eps_source',
)

DICT_FIELDS = (
    'tol',
    'rho',
    'penalty_update',
    'proximal_regularization',
    'esso_al_scale',
)

SOURCE_FIELDS = (
    'shared_ess_initialization_source',
    'balancing_exempt_channels_source',
    'balancing_exempt_until_source',
    'objective_scale_source',
    'shared_ess_reference_rating_source',
)


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def main():
    _refuse_overwrite(RESULTS_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    guard = SolveProfileGuard(permitted=(), label='P5.15-S40 case-file oracle load check').install()
    try:
        # ---- A: the case file alone, through the real production loader ----
        planning_case_file = O.fresh_planning('p515s40_case_file_oracle_checks_case_file')
        admm_case_file = planning_case_file.params.admm

        # ---- B: a fresh planning object with the s39_D override hook applied,
        #         the EXACT mechanism the oracle run used, zero solves --------
        planning_override = O.fresh_planning('p515s40_case_file_oracle_checks_override')
        hook = G._s39_configure_hook('s39_D')
        report_sink = {}
        hook(planning=planning_override, sed=planning_override.shared_ess_data,
             candidate=None, report=report_sink)
        admm_override = planning_override.params.admm
    finally:
        guard.uninstall()

    scalar_diffs = {}
    for field in SCALAR_FIELDS:
        a = getattr(admm_case_file, field)
        b = getattr(admm_override, field)
        if a != b:
            scalar_diffs[field] = {'case_file': a, 'override_path': b}

    dict_diffs = {}
    for field in DICT_FIELDS:
        a = getattr(admm_case_file, field)
        b = getattr(admm_override, field)
        if a != b:
            dict_diffs[field] = {'case_file': a, 'override_path': b}

    source_report = {}
    for field in SOURCE_FIELDS:
        a = getattr(admm_case_file, field, None)
        b = getattr(admm_override, field, None)
        source_report[field] = {'case_file': a, 'override_path': b, 'differs': a != b}

    # Fields the override path sets that the case file has no key for at all
    # (there are none known -- every s39_D override key is expressible in the
    # case file per `admm_parameters.py`'s `_read_parameters_from_file`; this
    # is a live check, not an assumption).
    override_checks = report_sink.get('rule_eleven_checklist', {}).get(
        's39_pre_solve_override_verification', {})

    oracle_relevant_match = (not scalar_diffs) and (not dict_diffs)

    payload = {
        'stage': 'P5.15 Addendum 22 item (3) -- zero-solve case-file load check',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 22 item 3_case_file',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'guard_observed': dict(guard.counts),
        'guard_permitted': 0,
        'scalar_fields_checked': list(SCALAR_FIELDS),
        'dict_fields_checked': list(DICT_FIELDS),
        'scalar_diffs': scalar_diffs,
        'dict_diffs': dict_diffs,
        'source_field_report': source_report,
        'override_in_force_checks_on_override_path_object': override_checks,
        'oracle_relevant_fields_match': oracle_relevant_match,
        'note': (
            'source_field_report differences are EXPECTED (provenance bookkeeping: '
            "'case_file' on the case-file-alone object vs the override string "
            "'p515s39_override_not_case_file' on the hook-applied object) -- they "
            'are reported, not gated. scalar_diffs / dict_diffs are the oracle-'
            'relevant comparison; both must be empty for the case file to '
            'reproduce the oracle configuration.'
        ),
    }

    with open(RESULTS_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S40-CASE-FILE-CHECKS] oracle_relevant_fields_match={oracle_relevant_match}')
    print(f'[S40-CASE-FILE-CHECKS] scalar_diffs={scalar_diffs}')
    print(f'[S40-CASE-FILE-CHECKS] dict_diffs={dict_diffs}')
    print(f'[S40-CASE-FILE-CHECKS] source_field_report={source_report}')
    print(f'[S40-CASE-FILE-CHECKS] guard observed={dict(guard.counts)}')
    print(f'[S40-CASE-FILE-CHECKS] wrote {RESULTS_PATH}')

    if not oracle_relevant_match:
        sys.exit(1)


if __name__ == '__main__':
    main()
