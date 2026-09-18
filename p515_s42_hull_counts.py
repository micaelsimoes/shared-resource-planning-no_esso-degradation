"""
P5.15 Addendum 24, Step 3 closure item -- hull-bound active counts per
channel EXCLUDING degenerate intervals.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 24: "Hull-bound counts are
to be reported excluding degenerate intervals (zero-solve, if cheap)" and
frozen spec v13's `reporting_obligations.hull_polish`: "hull-bound active
counts EXCLUDING degenerate intervals (zero-solve read of the committed
hull-polish evidence, if cheap)."

Zero-solve: reads the ALREADY-COMMITTED
`data/SRP1/Results/P515S41/hull_polish/hull_bound_detail.json` (commit
`2e6c5570`, 8,640 per-descriptor entries, each carrying `degenerate` and
`active` -- computed by `p515_s41_hull_polish._hull_bounds_active`, which
documents its own convention: "A degenerate entry counts as active by
definition"). The COMMITTED `hull_polish_results.json`'s own
`hull_bounds_active_by_channel` therefore INCLUDES every degenerate interval
as active; this script filters to `degenerate == False` first, so the
reported counts answer "how many GENUINELY constraining hull bounds were
active at the polished point", not "how many descriptors were either
degenerate or active". No re-run: this is cheap because the committed
artifact ALREADY carries the interval ends and the per-descriptor active
flag (spec's own parenthetical "if the committed artifacts carry the
interval ends"; they do -- `lo`/`hi`/`degenerate`/`active`/`polished_value`
per entry).

`SolveProfileGuard(permitted=())` armed for the whole script (no solve is
possible from a pure JSON read, but the guard is armed anyway per the task's
instruction and CLAUDE.md's "armed guards, never asserted" rule); `verify(0)`
checked before writing output.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s42_hull_counts.py
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s40_clone_capture_preflight import _sha256_file  # noqa: E402 -- BY IMPORT, unchanged

DETAIL_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S41', 'hull_polish', 'hull_bound_detail.json')
COMMITTED_RESULTS_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S41', 'hull_polish', 'hull_polish_results.json')

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'hull_counts')
OUT_PATH = os.path.join(OUT_DIR, 'hull_counts_excluding_degenerate.json')


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to reuse a non-fresh output dir: {OUT_DIR}')

    guard = SolveProfileGuard(permitted=(), label='P5.15-S42 hull counts').install()
    try:
        if not os.path.isfile(DETAIL_PATH):
            raise RuntimeError(
                f'committed hull-bound-detail artifact missing: {DETAIL_PATH} -- the '
                'per-descriptor interval ends (lo/hi/degenerate/active) are NOT '
                'available, so the excluding-degenerate count cannot be computed '
                'from committed evidence; it would need a re-run of '
                'p515_s41_hull_polish.py, which the spec forbids ("the hull polish '
                'is NOT re-run").')
        with open(DETAIL_PATH) as handle:
            descriptors = json.load(handle)

        total_all = Counter()
        active_all = Counter()  # matches the committed hull_polish_results.json convention
        total_non_degenerate = Counter()
        active_non_degenerate = Counter()
        degenerate_count = Counter()

        for d in descriptors:
            channel = d['channel']
            total_all[channel] += 1
            if d['active']:
                active_all[channel] += 1
            if d['degenerate']:
                degenerate_count[channel] += 1
            else:
                total_non_degenerate[channel] += 1
                if d['active']:
                    active_non_degenerate[channel] += 1

        channels = sorted(total_all)
        per_channel = {}
        for ch in channels:
            per_channel[ch] = {
                'n_total_descriptors': total_all[ch],
                'n_degenerate': degenerate_count[ch],
                'n_non_degenerate': total_non_degenerate[ch],
                'n_active_including_degenerate': active_all[ch],
                'n_active_excluding_degenerate': active_non_degenerate[ch],
                'active_excluding_degenerate_fraction_of_non_degenerate': (
                    active_non_degenerate[ch] / total_non_degenerate[ch]
                    if total_non_degenerate[ch] else None),
            }

        # Cross-check against the committed hull_polish_results.json's own
        # (degenerate-INCLUSIVE) counts, if present -- must match exactly, or
        # this script's re-derivation from hull_bound_detail.json disagrees
        # with the committed top-level summary and that is reported, not
        # silently reconciled.
        cross_check = None
        if os.path.isfile(COMMITTED_RESULTS_PATH):
            with open(COMMITTED_RESULTS_PATH) as handle:
                committed = json.load(handle)
            committed_active = committed.get('polish', {}).get('hull_bounds_active_by_channel', {})
            committed_total = committed.get('polish', {}).get('hull_bounds_total_by_channel', {})
            cross_check = {
                'committed_active_including_degenerate': committed_active,
                'recomputed_active_including_degenerate': dict(active_all),
                'committed_total': committed_total,
                'recomputed_total': dict(total_all),
                'matches': (
                    {k: committed_active.get(k) for k in channels}
                    == {k: active_all[k] for k in channels}
                    and {k: committed_total.get(k) for k in channels}
                    == {k: total_all[k] for k in channels}),
            }

    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(failures)

    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)
    out = {
        'stage': 'P5.15 Addendum 24 -- hull-bound active counts excluding degenerate intervals',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 24',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'source_artifact': os.path.relpath(DETAIL_PATH, REPO),
        'source_artifact_sha256': _sha256_file(DETAIL_PATH),
        'source_artifact_commit': '2e6c5570',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'n_descriptors_total': len(descriptors),
        'convention': (
            "'active' means: at the polished point, the constrained quantity sits within "
            "1e-9 relative of an interval end (p515_s41_hull_polish.HULL_ACTIVE_REL_TOL, "
            "scale max(|lo|,|hi|,1.0)). A degenerate interval (lo == hi) is counted as "
            "active BY DEFINITION in the committed hull_polish_results.json convention "
            "(_hull_bounds_active's own documented choice); this report's headline numbers "
            "are 'n_active_excluding_degenerate' / 'n_non_degenerate' -- degenerate entries "
            "removed from BOTH the numerator and the denominator."),
        'per_channel': per_channel,
        'cross_check_against_committed_results_json': cross_check,
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f'[S42-HULL-COUNTS] wrote {OUT_PATH}')

    manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
    manifest = {os.path.relpath(OUT_PATH, REPO): _sha256_file(OUT_PATH)}
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-HULL-COUNTS] wrote {manifest_path}')

    for ch, row in per_channel.items():
        print(f"[S42-HULL-COUNTS] {ch}: active_excl_degenerate={row['n_active_excluding_degenerate']}"
              f"/{row['n_non_degenerate']} (n_degenerate={row['n_degenerate']} of "
              f"{row['n_total_descriptors']} total)")
    if cross_check is not None:
        print(f"[S42-HULL-COUNTS] cross-check against committed hull_polish_results.json "
              f"matches={cross_check['matches']}")


if __name__ == '__main__':
    main()
