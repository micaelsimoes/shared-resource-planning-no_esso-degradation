"""
Stage P5.4-R0 -- canonical-environment provenance and fail-fast reproducibility gate.

Every P5.4-R harness calls `gate()` before doing any work. It records the full
runtime provenance and ABORTS unless EVERY canonical identity below matches. It
never silently continues.

The gate asserts four things, not one:

  * the realized SRP1 scenario checksum;
  * the IPOPT version actually invoked at `solver_params.solver_path`;
  * the ASL build stamped into that IPOPT banner;
  * the HSL linear solver configured in the network parameter files.

The last three were recorded but NOT asserted until the Mac Studio migration
(2026-09-09). That gap was not hypothetical: this machine initially resolved
IPOPT 3.14.20 built against ASL 20190605, which reproduces the canonical
checksum and would therefore have passed the checksum-only gate silently while
producing noncanonical numerical evidence. Recording a value the gate does not
compare is not a gate.

Note that the conda environment itself carries `conda-forge::ipopt 3.14.19`,
which is NOT the canonical solver. The canonical IPOPT is the standalone binary
at `/usr/local/bin/ipopt`, reached via `solver_params.solver_path`. The version
assertion is what keeps the two apart.

Canonical environment : /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python
Canonical checksum    : 5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358
Canonical IPOPT       : 3.14.18, ASL 20241111, at /usr/local/bin/ipopt
Canonical HSL solver  : ma97 (HSL 5.5.0; the library version is not machine-
                        readable from the IPOPT banner and is not asserted)

    python p54r_provenance.py          # run the gate on its own
"""

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CANONICAL_CHECKSUM = '5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358'
CANONICAL_ENV = 'opf_env_py311'

# Asserted alongside the checksum since the 2026-09-09 migration. See module
# docstring: a matching checksum under a different solver build is exactly the
# failure mode this exists to stop.
CANONICAL_IPOPT_VERSION = '3.14.18'
CANONICAL_IPOPT_ASL = '20241111'
CANONICAL_HSL_LINEAR_SOLVER = 'ma97'

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'


class ProvenanceError(RuntimeError):
    """Raised when the runtime is not the canonical paper environment."""


def _version(module_name):
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, '__version__', 'unknown')
    except Exception as error:
        return f'MISSING ({type(error).__name__})'


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def _ipopt_info(solver_path):
    info = {'path': solver_path, 'version': None, 'asl': None, 'resolved_from': None,
            'exists': bool(solver_path and os.path.exists(solver_path))}
    for candidate in (solver_path, 'ipopt'):
        if not candidate:
            continue
        try:
            out = subprocess.run([candidate, '--version'], capture_output=True,
                                 text=True, timeout=30)
            text = (out.stdout or '') + (out.stderr or '')
            match = re.search(r'Ipopt\s+([0-9]+\.[0-9]+\.[0-9]+)', text)
            if match:
                info['version'] = match.group(1)
                info['banner'] = text.strip().splitlines()[0][:200]
                info['resolved_from'] = candidate
                # e.g. "Ipopt 3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)"
                asl = re.search(r'ASL\s*\(\s*([0-9]{8})\s*\)', text)
                if asl:
                    info['asl'] = asl.group(1)
                break
        except Exception:
            continue
    return info


def _gurobi_info():
    info = {'gurobipy': None, 'gurobi_runtime': None, 'licence': None,
            'licence_expiry': None, 'available': False}
    try:
        import gurobipy as gp
        info['gurobipy'] = '.'.join(str(v) for v in gp.gurobi.version())
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(env=env)
        model.setParam('OutputFlag', 0)
        info['available'] = True
        info['gurobi_runtime'] = '.'.join(str(v) for v in gp.gurobi.version())
        model.dispose()
        env.dispose()
    except Exception as error:
        info['error'] = f'{type(error).__name__}: {error}'
    licence_path = os.path.expanduser('~/gurobi.lic')
    if os.path.exists(licence_path):
        fields = {}
        with open(licence_path) as handle:
            for line in handle:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key.upper() not in ('KEY', 'PASSWORD'):
                        fields[key.upper()] = value
        info['licence'] = fields.get('TYPE')
        info['licence_expiry'] = fields.get('EXPIRATION')
        info['licence_id'] = fields.get('LICENSEID')
        info['licence_version'] = fields.get('VERSION')
    return info


def collect(planning=None):
    """Collect full runtime provenance. Loads SRP1 if `planning` is not given."""
    console = io.StringIO()
    if planning is None:
        with redirect_stdout(console):
            from shared_resources_planning import SharedResourcesPlanning
            planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
            planning.read_planning_problem()
        setup_text = console.getvalue()
    else:
        setup_text = ''

    checksum = None
    for line in setup_text.splitlines():
        if 'checksum' in line.lower():
            checksum = line.split(':')[-1].strip()

    solver_params = planning.distribution_networks[
        list(planning.distribution_networks)[0]].params.solver_params
    solver_path = getattr(solver_params, 'solver_path', None)

    hsl = None
    try:
        hsl = solver_params.solver_options.get('linear_solver')
    except Exception:
        options = getattr(solver_params, 'options', None)
        if isinstance(options, dict):
            hsl = options.get('linear_solver')

    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git_head(),
        'sys_executable': sys.executable,
        'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'conda_env_from_path': (
            sys.executable.split('/envs/')[-1].split('/')[0]
            if '/envs/' in sys.executable else None),
        'python_version': sys.version.split()[0],
        'numpy_version': _version('numpy'),
        'pandas_version': _version('pandas'),
        'scipy_version': _version('scipy'),
        'pyomo_version': _version('pyomo'),
        'ipopt': _ipopt_info(solver_path),
        'hsl_linear_solver': hsl,
        'gurobi': _gurobi_info(),
        'scenario_checksum': checksum,
        'canonical_checksum': CANONICAL_CHECKSUM,
        'checksum_matches_canonical': checksum == CANONICAL_CHECKSUM,
        'canonical_ipopt_version': CANONICAL_IPOPT_VERSION,
        'canonical_ipopt_asl': CANONICAL_IPOPT_ASL,
        'canonical_hsl_linear_solver': CANONICAL_HSL_LINEAR_SOLVER,
    }, planning


def check(provenance):
    """The canonical identities the gate asserts. Returns a list of failures."""
    ipopt = provenance.get('ipopt') or {}
    comparisons = [
        ('scenario checksum', provenance.get('scenario_checksum'),
         CANONICAL_CHECKSUM),
        ('IPOPT version', ipopt.get('version'), CANONICAL_IPOPT_VERSION),
        ('IPOPT ASL build', ipopt.get('asl'), CANONICAL_IPOPT_ASL),
        ('HSL linear solver', provenance.get('hsl_linear_solver'),
         CANONICAL_HSL_LINEAR_SOLVER),
    ]
    return [{'identity': name, 'observed': observed, 'canonical': expected}
            for name, observed, expected in comparisons if observed != expected]


def gate(stage, out_dir, planning=None, verbose=True):
    """R0 fail-fast gate. Returns (provenance, planning); raises on mismatch."""
    provenance, planning = collect(planning)
    provenance['stage'] = stage
    failures = check(provenance)
    provenance['gate_failures'] = failures
    provenance['gate_passes'] = not failures
    os.makedirs(out_dir, exist_ok=True)
    # A run that fails the gate must never clobber a canonical provenance record.
    path = os.path.join(
        out_dir,
        'provenance.json' if provenance['gate_passes']
        else 'provenance_REJECTED.json')
    with open(path, 'w') as handle:
        json.dump(provenance, handle, indent=1, default=str)

    if verbose:
        print(f'[R0] provenance for {stage}')
        print(f"    executable      : {provenance['sys_executable']}")
        print(f"    CONDA_DEFAULT_ENV: {provenance['conda_default_env']} "
              f"(from path: {provenance['conda_env_from_path']})")
        print(f"    python {provenance['python_version']}  numpy {provenance['numpy_version']}  "
              f"pandas {provenance['pandas_version']}  scipy {provenance['scipy_version']}  "
              f"pyomo {provenance['pyomo_version']}")
        ipopt = provenance['ipopt']
        print(f"    IPOPT           : {ipopt.get('version')} ASL({ipopt.get('asl')}) "
              f"@ {ipopt.get('path')} (exists={ipopt.get('exists')})")
        print(f"    HSL solver      : {provenance['hsl_linear_solver']}")
        g = provenance['gurobi']
        print(f"    gurobipy        : {g.get('gurobipy')} available={g.get('available')} "
              f"licence={g.get('licence')} expires={g.get('licence_expiry')}")
        print(f"    checksum        : {provenance['scenario_checksum']}")
        print(f"    canonical match : {provenance['gate_passes']}")
        for failure in failures:
            print(f"      MISMATCH {failure['identity']}: "
                  f"observed {failure['observed']!r}, canonical {failure['canonical']!r}")
        print(f'    -> {path}')

    if failures:
        detail = '; '.join(
            f"{f['identity']} is {f['observed']!r}, canonical is {f['canonical']!r}"
            for f in failures)
        raise ProvenanceError(
            f"ABORT: the runtime is not the canonical paper environment. {detail}. "
            f"Running under {provenance['sys_executable']} with IPOPT at "
            f"{(provenance.get('ipopt') or {}).get('path')}. "
            f"Results are only valid in the canonical environment "
            f"({CANONICAL_ENV}). Not continuing."
        )
    return provenance, planning


if __name__ == '__main__':
    out = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R')
    try:
        gate('P5.4-R0 standalone', out)
        print('\n[R0] GATE PASSED — canonical environment confirmed.')
    except ProvenanceError as error:
        print(f'\n[R0] GATE FAILED\n{error}')
        sys.exit(1)
