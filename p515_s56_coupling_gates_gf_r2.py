"""
P5.15 S56 step 1 -- G-f smoke, SECOND ATTEMPT into a suffixed campaign root (launcher only; no gate logic here).

Why: the first G-f smoke under the frozen spec `frozen_coupling_gates_spec_v1_10c98258.json` (root
`campaign_s56cpl_gf_smoke`, verdict `verdict_gf_20261123T085107.json`) was refused by the child's rule-eleven
capture-path assertion BEFORE ANY SOLVE (exit 1 after 5 s): `assert_interface_coupling_capture_path` did not recognise
the eight `boyd_*_part` field names, which production composes as record[f'boyd_{group}_{part}']. The assertion was
fixed in the harness (committed separately). The frozen gate script hard-codes the root `s56cpl_gf_smoke`, which now
exists and is write-once, so the frozen spec cannot be followed as written; per the Planner's instruction the run goes
to a suffixed root and the spec is NOT edited.

This launcher imports the frozen gate script UNCHANGED (`p515_s56_coupling_gates.py`, sha256 recorded in the spec),
replaces only its module constant GF_ROOT with 's56cpl_gf_smoke_r2', and calls its own `smoke()`. Everything else --
configuration, options, cap, checks, verdict writer -- is the frozen script's.

Usage (repo root, canonical interpreter, attached, alone, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_coupling_gates_gf_r2.py --spec-sha256 <S> > <log> 2>&1
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s56_coupling_gates as G  # noqa: E402  (installs its zero-solve guard on this process, as when run directly)

GF_ROOT_R2 = 's56cpl_gf_smoke_r2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec-sha256', required=True)
    args = ap.parse_args()
    if os.environ.get('NLP_SOLVER_PATH') != G.SOLVER_PATH:
        G._log(f'REFUSING: NLP_SOLVER_PATH must be {G.SOLVER_PATH}')
        return 1
    if os.path.exists(os.path.join(G._gate_dir(), f'campaign_{GF_ROOT_R2}')):
        G._log(f'REFUSING: campaign_{GF_ROOT_R2} exists')
        return 1
    G._log(f'G-f second attempt: GF_ROOT {G.GF_ROOT!r} -> {GF_ROOT_R2!r} (frozen script otherwise unchanged)')
    G.GF_ROOT = GF_ROOT_R2
    return G.smoke(args)


if __name__ == '__main__':
    sys.exit(main())
