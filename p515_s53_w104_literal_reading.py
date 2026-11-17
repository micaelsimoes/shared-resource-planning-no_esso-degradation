"""
P5.15 Addendum 53, Planner task W104 -- ZERO SOLVES. Runs the COMMITTED W103 script `p515_s53_w103_literal_reading.py`
(unmodified; its sha256 is asserted below) on the SRP1 c_star continuation cell, writing to a NEW directory so that
W103's committed outputs are never touched.

The W103 module is imported (which arms its `SolveProfileGuard(permitted=())` for the whole run, verified at exactly 0
inside its `main`), then only its module-level configuration is re-pointed: `CELLS` -> the c_star cell alone (N 87,
CAP 187, P_MAX 22, asserted by W103's `cell_reading` against the frozen campaign spec), and `OUT_DIR` / `OUT` /
`MANIFEST` -> data/SRP1/Results/P515S53/w104_literal_reading/. Every computation, check (L, K1, K2, K3) and the
write-once refusal are W103's own code. W103's labels ('W103' in the stage string and the printed lines) are
therefore kept verbatim in the output; this wrapper is the record that the run is W104 on c_star.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w104_literal_reading.py \\
      > data/SRP1/Results/P515S53/w104_literal_reading/w104_literal_reading_launch.log 2>&1
"""
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

W103_SCRIPT = 'p515_s53_w103_literal_reading.py'
W103_SCRIPT_SHA256 = 'a7058ac96de5575c9d903c06443770b927816b881fbdf7c42ce061a752a1c3e0'  # as committed at 953b9bcd


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


assert _sha256(W103_SCRIPT) == W103_SCRIPT_SHA256, 'the W103 script differs from its committed version'

import p515_s53_w103_literal_reading as W103  # noqa: E402  (arms the zero-solve guard at import)

ROOT = W103.ROOT
W103.CELLS = {
    'c_star': {'campaign_spec': (f'{ROOT}/campaign_s53_w101_srp1_cont_c_star/'
                                 'campaign_spec_s53_w101_srp1_cont_c_star_1a7483ef.json'),
               'eval_dir': f'{ROOT}/campaign_s53_w101_srp1_cont_c_star/evals/4bf36c151fd10613_c_star',
               'campaign_results': f'{ROOT}/campaign_s53_w101_srp1_cont_c_star/campaign_results.json',
               'N_task': 87, 'CAP_task': 187},
}
W103.OUT_DIR = 'data/SRP1/Results/P515S53/w104_literal_reading'
W103.OUT = os.path.join(W103.OUT_DIR, 'w104_literal_reading.json')
W103.MANIFEST = os.path.join(W103.OUT_DIR, 'w104_literal_reading_manifest_sha256.json')

if __name__ == '__main__':
    print(f'[W104] running the committed {W103_SCRIPT} ({W103_SCRIPT_SHA256[:8]}) on cells {list(W103.CELLS)} -> '
          f'{W103.OUT_DIR}', flush=True)
    W103.main()
