"""
P5.15 Addendum 25 item 1 -- the Step 4 CAMPAIGN HARNESS.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 25 and 26; STEP4_DFO_METHOD.md
section 2 (oracle contract; 2.5 record, 2.7 interface); frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item1_harness`.

================================================================================
PROCESS MODEL
================================================================================
`evaluate(batch, ctx) -> records` (STEP4_DFO_METHOD.md 2.7). Every candidate
of the batch is evaluated in its OWN OS process -- a fresh interpreter
(`<python> -u p515_s44_campaign_harness.py --child ...`), never a fork of the
parent -- with the thread caps `THREAD_CAP_ENV` (OMP / MKL / OPENBLAS /
VECLIB / NUMEXPR = 1) placed in the child's environment before it starts (the
IPOPT executables it launches inherit them). At most `ctx.concurrency`
children run at once; the parent polls them (`os.wait4`, non-blocking, so
each child's own rusage is captured), fills free slots in batch order, and
returns one record per candidate, in batch order.

THE PARENT NEVER SOLVES: `main_parent`-level callers arm
`SolveProfileGuard(permitted=())` for the whole campaign and verify 0 at the
end (the gate script `p515_s44_gate.py` does). The parent imports no model
code on the evaluation path; it only spawns, watches and reads records.

EACH CHILD (one evaluation) runs the production oracle exactly as the D
oracle's case-file runs do (`p515_s43_aa_run.py` / `p515_s40_polish_gap.py`
pattern): `p515_g_g1_g4_admm_gates.run_admm_arm(label, eval_dir,
investment_map=<candidate>, num_max_iters_override=<spec cap>,
eval_id=<derived>, apply_rho=False, full_diagnostics_in_rows=True, ...)`
inside the SAME capture wrappers (`s38_pf_capture_hooks`,
`s39_exempt_until_capture_hooks`) and with the same terminal writer
(`write_boyd_terminal_s35ref`), so every artifact the D reference carries is
produced under the same names. Configuration = the case file alone; the
ONLY Python-side configuration action is `configuration.overrides` of the
frozen campaign spec, restricted to `SUPPORTED_OVERRIDE_KEYS` (empty for the
D configuration; the AA arm will name `anderson_acceleration`).

================================================================================
LOCKS
================================================================================
ONE campaign-level lock, `CAMPAIGN_LOCK_PATH` (`.p515_s44_campaign.lock`,
repo root), created O_EXCL by the parent for the whole campaign; content =
JSON {pid, campaign_id, campaign_spec_sha256, started_utc}; removed on exit.
The children NEVER take the legacy one-run lock (`.p515_g_gate.lock`,
`p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`) -- `run_admm_arm`
does not take it either (only the legacy harnesses' own `main`s do), so N
children run concurrently. Instead each child REFUSES TO START unless the
campaign lock exists, names ITS parent's pid (`os.getppid()`) and the same
campaign spec sha256 -- a child can run only under the live campaign that
spawned it. The parent refuses to start if EITHER lock exists. The legacy
lock is not taken by the parent (spec v14: the campaign lock replaces it for
campaign use). Since Addendum 25 item 2 (gate-ruling follow-up) the legacy
lock function `_acquire_exclusive_run_lock` refuses while the campaign lock
exists; both acquirers re-check the other lock AFTER creating their own, so a
simultaneous start cannot let both proceed.

Why concurrency is safe (checked, not assumed; see the worker report): each
evaluation has its own `eval_id` working dir (`p56a_oracle.fresh_planning`
isolates every holder's `logs_dir`, hence every IPOPT `output_file`, TSO,
DSO and ESSO), its own `results_dir` (`_set_results_dir_for_arm`, hence its
own FrozenSMOPF snapshots), its own output files, and pyomo's temporary NL/
SOL files are uniquely named; `ParallelExecution` is false and persistent
workers are off, so no process pool is shared.

================================================================================
WHAT EACH EVALUATION WRITES (write-once dir `<campaign_root>/evals/<key16>_<label>/`)
================================================================================
Parent: `launch.json` (command, env caps, start time, pid), `child_stdout.log`,
`child_stderr.log` (both streams), `exit_code.txt`, `wait4_rusage.json`.
Child: every `run_admm_arm` artifact (g_<label>.json with the full per-cycle
trajectory, heartbeat_<label>.json -- updated every ESSO solve --, stdout,
esso_capture/, leak/network-failure/recovery sidecars, esso_models pickle,
results/FrozenSMOPF), the capture sidecars (recourse-jump, ESS/PF entry
strides, SoH floor, ESS exempt-until state -- all appended EVERY cycle, so
they survive a crash), the terminal artifacts (boyd_terminal.json,
component_levels_terminal.json, interface_settlement_detail_s31c.json,
interface_voltage_terminal.json), `per_cycle_record.jsonl` (standard
per-cycle subset, derived from the trajectory), `evaluation_record.json`
(the STEP4 2.5 record, schema `RECORD_SCHEMA`), `child_manifest_sha256.json`.

Working-dir ids (`p56a_oracle.WORK_DIR/<id>`): `p515s44_<campaign_id>_<key16>_run`
and `..._precheck` -- derived from the campaign id and the canonical
candidate key; the child refuses if either exists (never reusable).

================================================================================
PER-EVALUATION CONFIGURATION AND THE POST-CERTIFICATION STEP (Addendum 25 item 2)
================================================================================
A spec entry is one EVALUATION (candidate x configuration). An entry may carry
its own `overrides` (replacing the campaign-level ones; ONLY
`anderson_acceleration.{enabled, reject_policy}` -- `validate_overrides`) and a
`post_certification` request (`resolve_post_certification`):
`persist_certified_models`, `hull_polish`, and `reference` = the D evaluation
of the SAME candidate (certified, no overrides; its record and component levels
hash-recorded in the spec at freeze time and re-verified by the child before the
run and before use). `eval_key` = the candidate key for the case-file
configuration, else sha256{candidate_key, overrides} (`evaluation_key`); it names
the eval dir and working-dir ids, so one campaign can hold C* under D and under AA.
AFTER the run, in the child, inside `run_admm_arm`'s post_run_hook (same live
models/state), `run_post_certification` does, only if the trajectory is
certified under the spec's bar (else it records `status: skipped` + reason):
(b) |Q - Q_ref| <= 1.5e-4 Q_ref and (c) the decomposition vs the reference
(`p515_s43_aa_run._cost_decomposition_vs_d(..., reference_dir=...)`: residual
<= 1.0, other priced components identically 0); then persists the certified
TSO/DSO models (`p515_s42_exact_fix_rerun._persist_certified_models`, BEFORE the
polish mutates them); then the interval-hull polish
(`p515_s41_hull_polish._polish_all_blocks_hull`: gate on the sum of per-block
changes, plus the settlement-excluded change, the settlement remainder and the
non-degenerate active-bound counts). Output: `post_certification.json`,
`hull_bound_detail.json`, `certified_models.pkl`, and a summary in the record.
An evaluation with AA on also gets `aa_per_cycle.jsonl`
(`p515_s43_aa_run._build_aa_per_cycle_sidecar`) and an action summary. A
post-certification exception is recorded (status 'error', traceback) and the
child exits 2 after writing its record; the evaluation itself stands.

================================================================================
FROZEN CAMPAIGN SPEC
================================================================================
`freeze_campaign_spec(...)` writes `<campaign_root>/campaign_spec_<id>_<sha8>.json`
(write-once; `<sha8>` = first 8 hex of the file's own sha256): campaign id,
candidates (label, canonical form, key), configuration (case file path,
sha256 and last commit; arm label; overrides), cap, concurrency, required
consecutive cycles, thread caps, interpreter, solver path, git HEAD, harness
sha256. Its sha256 is passed to every child, verified by the child against
the file, and written into every evaluation record.
"""

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HARNESS_PATH = os.path.abspath(__file__)
PYTHON = sys.executable
CAMPAIGN_LOCK_PATH = os.path.join(REPO, '.p515_s44_campaign.lock')
LEGACY_RUN_LOCK_PATH = os.path.join(REPO, '.p515_g_gate.lock')
RESULTS_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44')
CASE_FILE_REL = os.path.join('data', 'SRP1', 'SRP1_params.json')
CASE_FILE = os.path.join(REPO, CASE_FILE_REL)
RECORD_SCHEMA = 'p515_s44_evaluation_record_v2'  # v2 (Addendum 25 item 2): per-evaluation config + post-certification
SPEC_SCHEMA = 'p515_s44_campaign_spec_v2'

THREAD_CAP_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}
SUPPORTED_OVERRIDE_KEYS = frozenset({'anderson_acceleration'})
# Addendum 25 item 2: the ONLY configuration a campaign spec may override is the AA flag and its
# reject-policy sub-option (memory / regularization stay at the frozen 5 / 1e-10).
SUPPORTED_AA_OVERRIDE_SUBKEYS = frozenset({'enabled', 'reject_policy'})
FROZEN_AA_MEMORY = 5
FROZEN_AA_REGULARIZATION = 1e-10
POST_CERTIFICATION_KEYS = frozenset({'persist_certified_models', 'hull_polish', 'reference'})
EVALUATION_OPTION_KEYS = frozenset({'overrides', 'post_certification'})
ACTIVE_NODES = (5, 7, 9)
INVESTMENT_YEAR = 2025
BAR_WINDOW = 10  # STEP4 2.5: "its bar (max objective step over the last 10 cycles)"
CHANNELS = ('v', 'pf', 'ess')

POLL_S = 5.0
HEARTBEAT_EVERY_S = 60.0

PER_CYCLE_RECORD_FIELDS = (
    'cycle', 'local_solves_ok', 'recourse', 'gross_operational_cost', 'terminal_salvage_value',
    'objective_change_abs', 'objective_tolerance', 'objective_change_ratio',
    'cycle_convergence', 'consecutive_converged_cycles', 'boyd_all_pass', 'boyd_stop',
    'boyd_v_primal_ratio', 'boyd_v_dual_ratio', 'boyd_v_channel_pass',
    'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio', 'boyd_pf_channel_pass',
    'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_ess_channel_pass',
    'rho_v_after', 'rho_pf_after', 'rho_ess_after', 'rho_v_action', 'rho_pf_action', 'rho_ess_action',
    'rho_freeze_active', 'efc_per_day_max',
)


# ==============================================================================
#  small utilities
# ==============================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(obj, handle, indent=1, default=str)
    os.replace(tmp, path)


def _write_once_json(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=1, default=str)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def _sanitize_id(text):
    out = ''.join(c if (c.isalnum() or c == '_') else '_' for c in str(text).lower())
    if not out:
        raise ValueError(f'empty id after sanitization: {text!r}')
    return out


# ==============================================================================
#  candidates: canonical form and key
# ==============================================================================
def canonical_candidate(candidate, active_nodes=ACTIVE_NODES, investment_year=INVESTMENT_YEAR):
    """Canonical per-node (s_mva, e_mwh) investment map at one investment year.

    `candidate` is {node_id: (s_mva, e_mwh)} (node ids int or str). Every active
    node must be present (no implicit zeros -- `run_admm_arm`'s own docstring:
    "pass a full dict covering every active node to avoid ambiguity"); values
    are floats >= 0; s == 0 <=> e == 0 (production raises otherwise,
    `_configure_esso_cohort_state`). Canonical form: {'investment_year': y,
    'nodes': {'5': [s, e], ...}} with float values and string node keys in
    ascending node order -- two candidates with the same canonical form are
    the same evaluation (STEP4 5.4)."""
    if not isinstance(candidate, dict):
        raise TypeError(f'candidate must be a dict {{node: (s, e)}}, got {type(candidate).__name__}')
    by_node = {}
    for key, value in candidate.items():
        node = int(key)
        if node in by_node:
            raise ValueError(f'node {node} given twice')
        s_val, e_val = value
        s_val, e_val = float(s_val), float(e_val)
        for name, v in (('s', s_val), ('e', e_val)):
            if not (v == v) or v in (float('inf'), float('-inf')):
                raise ValueError(f'node {node}: non-finite {name}={v}')
            if v < 0.0:
                raise ValueError(f'node {node}: negative {name}={v}')
        if (s_val == 0.0) != (e_val == 0.0):
            raise ValueError(f'node {node}: s == 0 <=> e == 0 violated (s={s_val}, e={e_val})')
        by_node[node] = (s_val + 0.0, e_val + 0.0)  # +0.0 folds -0.0 into 0.0
    missing = sorted(set(active_nodes) - set(by_node))
    extra = sorted(set(by_node) - set(active_nodes))
    if missing or extra:
        raise ValueError(f'candidate must cover exactly the active nodes {list(active_nodes)}: '
                         f'missing={missing} extra={extra}')
    return {'investment_year': int(investment_year),
            'nodes': {str(n): [by_node[n][0], by_node[n][1]] for n in sorted(by_node)}}


def candidate_key(canonical):
    """sha256 (hex) of the canonical form's compact, key-sorted JSON (floats by repr)."""
    text = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def investment_map_from_canonical(canonical):
    return {int(n): (v[0], v[1]) for n, v in canonical['nodes'].items()}


def eval_ids(campaign_id, key):
    stub = f'p515s44_{_sanitize_id(campaign_id)}_{key[:16]}'
    return {'run': f'{stub}_run', 'precheck': f'{stub}_precheck'}


# ==============================================================================
#  per-evaluation configuration (Addendum 25 item 2)
# ==============================================================================
def validate_overrides(overrides):
    """The spec's configuration overrides: ONLY `anderson_acceleration` with
    sub-keys `enabled` (bool) and `reject_policy` (one of
    `admm_anderson_acceleration.REJECT_POLICIES`). Returns a normalized copy."""
    overrides = dict(overrides or {})
    unsupported = sorted(set(overrides) - SUPPORTED_OVERRIDE_KEYS)
    if unsupported:
        raise ValueError(f'unsupported configuration overrides {unsupported}; supported: '
                         f'{sorted(SUPPORTED_OVERRIDE_KEYS)}')
    out = {}
    if 'anderson_acceleration' in overrides:
        import admm_anderson_acceleration as AA  # numpy only; no model code
        aa = overrides['anderson_acceleration']
        if not isinstance(aa, dict):
            raise ValueError('anderson_acceleration override must be a dict')
        bad = sorted(set(aa) - SUPPORTED_AA_OVERRIDE_SUBKEYS)
        if bad:
            raise ValueError(f'unsupported anderson_acceleration override sub-keys {bad}; supported: '
                             f'{sorted(SUPPORTED_AA_OVERRIDE_SUBKEYS)}')
        if 'enabled' in aa and not isinstance(aa['enabled'], bool):
            raise ValueError('anderson_acceleration.enabled must be a bool')
        if 'reject_policy' in aa and aa['reject_policy'] not in AA.REJECT_POLICIES:
            raise ValueError(f"anderson_acceleration.reject_policy must be one of {AA.REJECT_POLICIES}")
        out['anderson_acceleration'] = dict(aa)
    return out


def evaluation_key(candidate_key_hex, overrides):
    """Identity of one EVALUATION (candidate x configuration). The case-file
    configuration (no overrides) keeps the candidate key itself, so a D
    evaluation's directory name is `<candidate key16>_<label>` as in s44_gate;
    any override gives sha256 of {candidate_key, overrides}."""
    if not overrides:
        return candidate_key_hex
    text = json.dumps({'candidate_key': candidate_key_hex, 'overrides': overrides}, sort_keys=True,
                      separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_post_certification(request, candidate_key_hex):
    """Validate a per-evaluation post-certification request and resolve its
    reference evaluation (hash-recorded) at spec-freeze time. Returns None if
    nothing is requested. The reference must be a CERTIFIED evaluation of the
    SAME candidate under the case-file configuration (no overrides), i.e. the D
    evaluation of that candidate."""
    if not request:
        return None
    if not isinstance(request, dict):
        raise ValueError('post_certification must be a dict')
    bad = sorted(set(request) - POST_CERTIFICATION_KEYS)
    if bad:
        raise ValueError(f'unsupported post_certification keys {bad}; supported: {sorted(POST_CERTIFICATION_KEYS)}')
    persist = bool(request.get('persist_certified_models', False))
    polish = bool(request.get('hull_polish', False))
    for name in ('persist_certified_models', 'hull_polish'):
        if name in request and not isinstance(request[name], bool):
            raise ValueError(f'post_certification.{name} must be a bool')
    ref = request.get('reference')
    resolved_ref = None
    if ref is not None:
        if not isinstance(ref, dict) or set(ref) != {'eval_dir'}:
            raise ValueError("post_certification.reference must be {'eval_dir': <repo-relative eval dir>}")
        ref_dir = os.path.join(REPO, ref['eval_dir'])
        rec_path = os.path.join(ref_dir, 'evaluation_record.json')
        cl_path = os.path.join(ref_dir, 'component_levels_terminal.json')
        for p in (rec_path, cl_path):
            if not os.path.isfile(p):
                raise ValueError(f'reference evaluation file missing: {p}')
        with open(rec_path) as handle:
            ref_rec = json.load(handle)
        with open(cl_path) as handle:
            ref_cl = json.load(handle)
        ref_overrides = (ref_rec.get('evaluation_overrides_effective')
                         if 'evaluation_overrides_effective' in ref_rec
                         else (ref_rec.get('configuration') or {}).get('overrides'))
        problems = []
        if ref_rec.get('status') != 'certified' or ref_rec.get('certified_cost') is None:
            problems.append(f"reference not certified (status={ref_rec.get('status')})")
        if ref_rec.get('candidate_key') != candidate_key_hex:
            problems.append(f"reference candidate_key {str(ref_rec.get('candidate_key'))[:16]} != "
                            f'{candidate_key_hex[:16]} (must be the SAME candidate)')
        if ref_overrides:
            problems.append(f'reference is not the case-file (D) configuration: overrides={ref_overrides}')
        ref_gross = (ref_cl.get('recourse_components') or {}).get('gross_operational_cost')
        if ref_gross != ref_rec.get('certified_cost'):
            problems.append(f"reference component_levels gross {ref_gross} != record certified_cost "
                            f"{ref_rec.get('certified_cost')}")
        if problems:
            raise ValueError(f'invalid post_certification.reference {ref["eval_dir"]}: {problems}')
        resolved_ref = {
            'eval_dir': ref['eval_dir'],
            'evaluation_record_sha256': sha256_file(rec_path),
            'component_levels_terminal_sha256': sha256_file(cl_path),
            'certified_cost': ref_rec.get('certified_cost'),
            'certification_cycle': ref_rec.get('certification_cycle'),
            'candidate_key': ref_rec.get('candidate_key'),
            'campaign_spec_sha256': ref_rec.get('campaign_spec_sha256'),
            'configuration_overrides': ref_overrides or {},
        }
    if not (persist or polish or resolved_ref):
        return None
    return {'persist_certified_models': persist, 'hull_polish': polish, 'reference': resolved_ref}


def verify_reference_unchanged(resolved_ref):
    """Child side (before the run, and again before use): the reference files
    still hash to what the frozen spec recorded."""
    ref_dir = os.path.join(REPO, resolved_ref['eval_dir'])
    got = {'evaluation_record_sha256': sha256_file(os.path.join(ref_dir, 'evaluation_record.json')),
           'component_levels_terminal_sha256': sha256_file(os.path.join(ref_dir, 'component_levels_terminal.json'))}
    bad = {k: (resolved_ref[k], v) for k, v in got.items() if resolved_ref[k] != v}
    if bad:
        raise RuntimeError(f'post-certification reference changed since the spec was frozen: {bad}')
    return got


def eval_dir_name(key, label):
    return f'{key[:16]}_{_sanitize_id(label)}'


# ==============================================================================
#  frozen campaign spec
# ==============================================================================
def freeze_campaign_spec(campaign_root, campaign_id, candidates, configuration, cap, concurrency,
                         authority, required_consecutive_cycles=10, extra=None):
    """Write the campaign's frozen spec (write-once) and return (path, sha256, spec).

    `candidates`: list of (label, {node: (s, e)}) or (label, {node: (s, e)}, options).
    `options` (Addendum 25 item 2) may carry
      - 'overrides': this evaluation's configuration overrides, REPLACING the
        campaign-level `configuration['overrides']` for it (validated by
        `validate_overrides`: the AA flag and its reject-policy only);
      - 'post_certification': {'persist_certified_models': bool, 'hull_polish':
        bool, 'reference': {'eval_dir': ...} | None} (`resolve_post_certification`).
    One entry = one EVALUATION: its `eval_key` (`evaluation_key`) identifies
    candidate x configuration; labels and eval keys must be unique (the same
    candidate may appear under two configurations)."""
    if os.path.exists(campaign_root) and os.listdir(campaign_root):
        raise RuntimeError(f'campaign root exists and is not empty (write-once): {campaign_root}')
    overrides = validate_overrides(configuration.get('overrides'))
    cand_entries, seen_labels, seen_keys = [], set(), set()
    for item in candidates:
        if len(item) == 2:
            (label, cand), options = item, {}
        else:
            label, cand, options = item
            options = dict(options or {})
        bad = sorted(set(options) - EVALUATION_OPTION_KEYS)
        if bad:
            raise ValueError(f'unsupported evaluation options {bad} for {label}; supported: '
                             f'{sorted(EVALUATION_OPTION_KEYS)}')
        canon = canonical_candidate(cand)
        key = candidate_key(canon)
        eff_overrides = validate_overrides(options['overrides']) if 'overrides' in options else dict(overrides)
        ekey = evaluation_key(key, eff_overrides)
        if label in seen_labels or ekey in seen_keys:
            raise ValueError(f'duplicate evaluation label or key (candidate x configuration): {label} / {ekey[:16]}')
        seen_labels.add(label)
        seen_keys.add(ekey)
        post_cert = resolve_post_certification(options.get('post_certification'), key)
        cand_entries.append({'label': label, 'canonical': canon, 'key': key, 'eval_key': ekey,
                             'overrides': eff_overrides, 'post_certification': post_cert,
                             'eval_dir': eval_dir_name(ekey, label),
                             'working_dir_ids': eval_ids(campaign_id, ekey)})
    os.makedirs(campaign_root, exist_ok=True)  # only after every validation above has passed
    try:
        head = _git(['rev-parse', 'HEAD'])
        case_file_commit = _git(['log', '-1', '--format=%H', '--', CASE_FILE_REL])
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f'git provenance unavailable: {error}') from error
    spec = {
        'schema': SPEC_SCHEMA,
        'campaign_id': campaign_id,
        'frozen_utc': _utc(),
        'authority': list(authority),
        'candidates': cand_entries,
        'configuration': {
            'name': configuration['name'],
            'arm_label': configuration.get('arm_label', 's39_D'),
            'case_file': CASE_FILE_REL,
            'case_file_sha256': sha256_file(CASE_FILE),
            'case_file_last_commit': case_file_commit,
            'overrides': overrides,
            'apply_rho': False,
            'full_diagnostics_in_rows': True,
            'note': configuration.get('note'),
        },
        'cap': int(cap),
        'concurrency': int(concurrency),
        'required_consecutive_cycles': int(required_consecutive_cycles),
        'bar_window_cycles': BAR_WINDOW,
        'thread_caps': dict(THREAD_CAP_ENV),
        'interpreter': PYTHON,
        'nlp_solver_path_env': _resolve_solver_path_from_dotenv(),
        'git_head': head,
        'harness': {'path': os.path.relpath(HARNESS_PATH, REPO), 'sha256': sha256_file(HARNESS_PATH)},
        'extra': extra or {},
    }
    text = json.dumps(spec, indent=1, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(campaign_root, f'campaign_spec_{_sanitize_id(campaign_id)}_{digest[:8]}.json')
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite frozen campaign spec: {path}')
    with open(path, 'w') as handle:
        handle.write(text)
    if sha256_file(path) != digest:
        raise RuntimeError('frozen spec hash mismatch after write')
    return path, digest, spec


def _resolve_solver_path_from_dotenv():
    """What the children will resolve for NLP_SOLVER_PATH (solver_parameters.py loads .env)."""
    value = os.environ.get('NLP_SOLVER_PATH')
    if value:
        return {'NLP_SOLVER_PATH': value, 'source': 'environment'}
    env_path = os.path.join(REPO, '.env')
    if os.path.exists(env_path):
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith('NLP_SOLVER_PATH='):
                    return {'NLP_SOLVER_PATH': line.split('=', 1)[1].strip().strip('"\''), 'source': '.env'}
    return {'NLP_SOLVER_PATH': None, 'source': 'unresolved'}


def load_frozen_spec(campaign_root, expected_sha256):
    hits = [f for f in os.listdir(campaign_root) if f.startswith('campaign_spec_') and f.endswith('.json')]
    matches = [f for f in hits if sha256_file(os.path.join(campaign_root, f)) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one frozen spec with sha256 {expected_sha256} in '
                           f'{campaign_root}; found {matches} among {hits}')
    path = os.path.join(campaign_root, matches[0])
    with open(path) as handle:
        return path, json.load(handle)


# ==============================================================================
#  the campaign lock
# ==============================================================================
def acquire_campaign_lock(campaign_id, spec_sha256, lock_path=CAMPAIGN_LOCK_PATH,
                          legacy_lock_path=LEGACY_RUN_LOCK_PATH):
    if os.path.exists(legacy_lock_path):
        raise SystemExit(f'REFUSING TO RUN: the legacy one-run lock exists: {legacy_lock_path}')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = open(lock_path).read().strip()
        except OSError:
            holder = 'unknown'
        raise SystemExit(f'REFUSING TO RUN: campaign lock held: {lock_path} ({holder})')
    content = {'pid': os.getpid(), 'campaign_id': campaign_id, 'campaign_spec_sha256': spec_sha256,
               'started_utc': _utc()}
    os.write(fd, json.dumps(content).encode())
    os.close(fd)
    if os.path.exists(legacy_lock_path):
        # Mirror of `p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`'s
        # post-create re-check (Addendum 25 item 2 follow-up): a legacy one-run
        # lock appeared between the first check and our O_EXCL create -- back
        # out (remove OUR lock) and refuse, so the two can never both proceed.
        os.remove(lock_path)
        raise SystemExit(f'REFUSING TO RUN: the legacy one-run lock appeared: {legacy_lock_path}')
    return content


def release_campaign_lock(lock_path=CAMPAIGN_LOCK_PATH, expected_pid=None):
    if not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path) as handle:
            content = json.load(handle)
    except (OSError, ValueError):
        content = {}
    if expected_pid is not None and content.get('pid') != expected_pid:
        raise RuntimeError(f'campaign lock {lock_path} is not held by pid {expected_pid}: {content}')
    os.remove(lock_path)
    return True


def verify_child_lock(spec_sha256, lock_path=CAMPAIGN_LOCK_PATH):
    """Child side: the campaign lock must exist, name THIS child's parent and the same spec."""
    if not os.path.exists(lock_path):
        raise SystemExit(f'CHILD REFUSES: no campaign lock at {lock_path}')
    with open(lock_path) as handle:
        content = json.load(handle)
    if content.get('pid') != os.getppid():
        raise SystemExit(f'CHILD REFUSES: campaign lock pid {content.get("pid")} != parent pid {os.getppid()}')
    if content.get('campaign_spec_sha256') != spec_sha256:
        raise SystemExit('CHILD REFUSES: campaign lock spec sha256 differs from the spec this child was given')
    return content


# ==============================================================================
#  preconditions (parent)
# ==============================================================================
PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'shared_resources_planning.py', 'network.py', 'network_data.py', 'shared_energy_storage_data.py',
    'admm_parameters.py', 'admm_anderson_acceleration.py', 'admm_persistent_workers.py',
    'model_construction_helpers.py', 'p515_g_g1_g4_admm_gates.py', 'p56a_oracle.py',
    'p514_n_instrumented_cstar.py', 'p515_s39_evaluate.py', 'p515_s40_polish_gap.py',
    'p515_s40_clone_capture_preflight.py', 'p515_s43_aa_flagoff_gate.py',
    'p515_s44_campaign_harness.py', CASE_FILE_REL,
    # Addendum 25 item 2: imported by the post-certification step / AA sidecar
    'p515_s41_hull_polish.py', 'p515_s42_exact_fix_rerun.py', 'p515_s43_aa_run.py',
    'p515_s40_cost_decomposition.py',
)
FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s4')


def _ancestor_pids(max_depth=15):
    """This process and its ancestors (same reasoning as
    `p515_s40_clone_capture_preflight._ancestor_pids`, which is not imported
    here so the parent stays free of model imports)."""
    pids = {os.getpid()}
    current = os.getpid()
    for _ in range(max_depth):
        try:
            ppid_text = subprocess.run(['ps', '-o', 'ppid=', '-p', str(current)], capture_output=True,
                                       text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            break
        if not ppid_text:
            break
        ppid = int(ppid_text)
        if ppid <= 1 or ppid in pids:
            break
        pids.add(ppid)
        current = ppid
    return pids


def check_campaign_preconditions(campaign_root, extra_clean_files=(), lock_path=CAMPAIGN_LOCK_PATH,
                                 legacy_lock_path=LEGACY_RUN_LOCK_PATH):
    failures = []
    if os.path.exists(legacy_lock_path):
        failures.append(f'legacy one-run lock exists: {legacy_lock_path}')
    if os.path.exists(lock_path):
        failures.append(f'campaign lock exists: {lock_path}')
    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded = {str(p) for p in _ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded:
            continue
        if any(s in line for s in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')
    if os.path.exists(campaign_root):
        failures.append(f'campaign root already exists (write-once): {campaign_root}')
    try:
        status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN)
                      + list(extra_clean_files))
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production/harness files not clean in git:\n{status}')
    return failures


# ==============================================================================
#  evaluate(batch) -- the parent side
# ==============================================================================
class CampaignContext:
    def __init__(self, campaign_root, spec_path, spec_sha256, spec, log=print, child_extra_args=()):
        self.campaign_root = campaign_root
        self.spec_path = spec_path
        self.spec_sha256 = spec_sha256
        self.spec = spec
        self.concurrency = int(spec['concurrency'])
        self.log = log
        self.child_extra_args = tuple(child_extra_args)  # checks-only (stub mode); never set by a campaign
        self.evals_root = os.path.join(campaign_root, 'evals')


def _entry_eval_key(entry):
    return entry.get('eval_key', entry['key'])  # v1 specs (s44_gate) carry no eval_key: eval key == candidate key


def _spec_candidate(ctx, candidate):
    """Resolve one batch item to its spec entry: a str is an evaluation LABEL;
    a dict is a candidate, which must then match exactly one entry (a candidate
    listed under two configurations must be named by label)."""
    if isinstance(candidate, str):
        hits = [e for e in ctx.spec['candidates'] if e['label'] == candidate]
        if len(hits) != 1:
            raise ValueError(f'evaluation label {candidate!r} is not (uniquely) in the frozen campaign spec')
        return hits[0]
    canon = canonical_candidate(candidate)
    key = candidate_key(canon)
    hits = [e for e in ctx.spec['candidates'] if e['key'] == key]
    if len(hits) > 1:
        raise ValueError(f'candidate {key[:16]} appears under {len(hits)} configurations; name it by label')
    if hits:
        return hits[0]
    raise ValueError(f'candidate {canon} (key {key[:16]}) is not in the frozen campaign spec')


def _child_command(ctx, entry):
    return [PYTHON, '-u', HARNESS_PATH, '--child', '--campaign-root', ctx.campaign_root,
            '--spec-sha256', ctx.spec_sha256, '--eval-key', _entry_eval_key(entry)] + list(ctx.child_extra_args)


def _rusage_dict(ru):
    return {k: getattr(ru, k) for k in ('ru_utime', 'ru_stime', 'ru_maxrss', 'ru_minflt', 'ru_majflt',
                                         'ru_nvcsw', 'ru_nivcsw')}


def _tail(path, n=40):
    try:
        with open(path, errors='replace') as handle:
            return handle.readlines()[-n:]
    except OSError:
        return []


def _barrier_record_for_missing(ctx, entry, eval_dir, exit_code):
    return {
        'schema': RECORD_SCHEMA,
        'campaign_id': ctx.spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(ctx.spec_path, REPO),
        'campaign_spec_sha256': ctx.spec_sha256,
        'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'], 'eval_key': _entry_eval_key(entry),
        'status': 'harness_error', 'barrier': True,
        'barrier_cause': f'child exited with code {exit_code} and wrote no evaluation_record.json',
        'stderr_tail': _tail(os.path.join(eval_dir, 'child_stderr.log')),
        'stdout_tail': _tail(os.path.join(eval_dir, 'child_stdout.log')),
        'synthesized_by_parent': True,
    }


def evaluate(batch, ctx):
    """STEP4_DFO_METHOD.md 2.7: `evaluate(batch: list[x]) -> list[record]`.

    `batch`: list of candidates ({node: (s, e)}) or evaluation labels (str),
    each present in the frozen campaign spec, no duplicates (a candidate listed
    under two configurations must be given by label). Returns the evaluation records in batch
    order (a parent-synthesized barrier record when a child left none)."""
    entries = [_spec_candidate(ctx, x) for x in batch]
    keys = [_entry_eval_key(e) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate evaluations in one batch')
    os.makedirs(ctx.evals_root, exist_ok=True)
    for entry in entries:
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        if os.path.exists(eval_dir):
            raise RuntimeError(f'evaluation dir already exists (write-once, never reusable): {eval_dir}')
    pending = list(range(len(entries)))
    running = {}  # pid -> (index, proc, started, handles)
    results = [None] * len(entries)
    child_env = dict(os.environ)
    child_env.update(THREAD_CAP_ENV)
    max_concurrent_observed = 0
    last_heartbeat = 0.0
    timeline = []

    def _launch(i):
        entry = entries[i]
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        os.makedirs(eval_dir)
        out_h = open(os.path.join(eval_dir, 'child_stdout.log'), 'w')
        err_h = open(os.path.join(eval_dir, 'child_stderr.log'), 'w')
        cmd = _child_command(ctx, entry)
        proc = subprocess.Popen(cmd, cwd=REPO, env=child_env, stdout=out_h, stderr=err_h,
                                stdin=subprocess.DEVNULL)
        started = time.time()
        _write_once_json(os.path.join(eval_dir, 'launch.json'), {
            'command': cmd, 'cwd': REPO, 'thread_caps_in_child_env': dict(THREAD_CAP_ENV),
            'pid': proc.pid, 'parent_pid': os.getpid(), 'started_utc': _utc(),
            'campaign_spec_sha256': ctx.spec_sha256, 'label': entry['label'], 'key': entry['key'],
            'eval_key': _entry_eval_key(entry),
        })
        running[proc.pid] = (i, proc, started, (out_h, err_h))
        timeline.append({'event': 'start', 'label': entry['label'], 'pid': proc.pid, 't': started})
        ctx.log(f'[S44-HARNESS] launched {entry["label"]} (key {entry["key"][:16]}) pid={proc.pid}')

    def _reap(pid, status, ru):
        i, proc, started, handles = running.pop(pid)
        for h in handles:
            h.close()
        exit_code = os.waitstatus_to_exitcode(status)
        proc.returncode = exit_code  # reaped via os.wait4; keep Popen consistent
        ended = time.time()
        entry = entries[i]
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        with open(os.path.join(eval_dir, 'exit_code.txt'), 'w') as handle:
            handle.write(f'{exit_code}\n')
        _write_once_json(os.path.join(eval_dir, 'wait4_rusage.json'), {
            'rusage': _rusage_dict(ru), 'ru_maxrss_units': 'bytes on macOS/BSD, kilobytes on Linux',
            'semantics': 'os.wait4 rusage of the child process as reaped by the parent',
            'wall_s_parent_view': ended - started, 'exit_code': exit_code,
        })
        record_path = os.path.join(eval_dir, 'evaluation_record.json')
        if os.path.exists(record_path):
            with open(record_path) as handle:
                record = json.load(handle)
        else:
            record = _barrier_record_for_missing(ctx, entry, eval_dir, exit_code)
            _write_once_json(os.path.join(eval_dir, 'parent_barrier_record.json'), record)
        record = dict(record)
        record['parent_view'] = {'exit_code': exit_code, 'wall_s': ended - started,
                                 'wait4_ru_maxrss': ru.ru_maxrss}
        results[i] = record
        timeline.append({'event': 'end', 'label': entry['label'], 'pid': pid, 't': ended,
                         'exit_code': exit_code})
        ctx.log(f'[S44-HARNESS] finished {entry["label"]} pid={pid} exit={exit_code} '
                f'status={record.get("status")} wall={ended - started:.0f}s')

    def _heartbeat():
        status = []
        for pid, (i, _p, started, _h) in running.items():
            entry = entries[i]
            eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
            hb_path = os.path.join(eval_dir, f"heartbeat_{ctx.spec['configuration']['arm_label']}.json")
            hb = None
            if os.path.exists(hb_path):
                try:
                    with open(hb_path) as handle:
                        hb = json.load(handle)
                except (OSError, ValueError):
                    hb = 'unreadable (being written)'
            status.append({'label': entry['label'], 'pid': pid, 'elapsed_s': time.time() - started,
                           'production_heartbeat': hb})
        _atomic_write_json(os.path.join(ctx.campaign_root, 'campaign_heartbeat.json'), {
            'utc': _utc(), 'parent_pid': os.getpid(), 'running': status,
            'pending': [entries[i]['label'] for i in pending],
            'done': [entries[i]['label'] for i, r in enumerate(results) if r is not None]})
        ctx.log('[S44-HARNESS] heartbeat ' + '; '.join(
            f"{s['label']}: {s['elapsed_s']:.0f}s cycle="
            f"{s['production_heartbeat'].get('cycle') if isinstance(s['production_heartbeat'], dict) else None}"
            for s in status))

    try:
        while pending or running:
            while pending and len(running) < ctx.concurrency:
                _launch(pending.pop(0))
            max_concurrent_observed = max(max_concurrent_observed, len(running))
            for pid in list(running):
                reaped_pid, status, ru = os.wait4(pid, os.WNOHANG)
                if reaped_pid == pid:
                    _reap(pid, status, ru)
            now = time.time()
            if running and now - last_heartbeat >= HEARTBEAT_EVERY_S:
                _heartbeat()
                last_heartbeat = now
            if running or pending:
                time.sleep(POLL_S)
    except BaseException:
        for pid, (_i, proc, _s, _h) in list(running.items()):
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in list(running):
            try:
                _rp, status, ru = os.wait4(pid, 0)
                _reap(pid, status, ru)
            except ChildProcessError:
                running.pop(pid, None)
        raise
    _atomic_write_json(os.path.join(ctx.campaign_root, 'campaign_heartbeat.json'), {
        'utc': _utc(), 'parent_pid': os.getpid(), 'running': [], 'pending': [],
        'done': [e['label'] for e in entries]})
    evaluate.last_batch_info = {'max_concurrent_observed': max_concurrent_observed, 'timeline': timeline}
    return results


# ==============================================================================
#  the STEP4 2.5 record (pure function of an evaluation's own artifacts)
# ==============================================================================
def _max_step_last_n(rows, n=BAR_WINDOW):
    tail = rows[-n:] if len(rows) >= n else rows
    steps = [(r.get('cycle'), r.get('objective_change_abs')) for r in tail]
    vals = [s for _c, s in steps if s is not None]
    return {'definition': f'max objective_change_abs (|gross cost step|) over the last {n} cycles',
            'value': max(vals) if vals else None, 'n_cycles_in_window': len(tail),
            'n_steps_available': len(vals), 'window': [{'cycle': c, 'objective_change_abs': s} for c, s in steps]}


def _storage_per_node(canonical, esso_capture, floor_terminal, published_caps):
    out = {}
    per_node_floor = {}
    if floor_terminal and floor_terminal.get('available'):
        for e in floor_terminal.get('per_node_per_cohort_year') or []:
            per_node_floor.setdefault(str(e.get('node_id')), []).append(e)
    for node, (s_val, e_val) in canonical['nodes'].items():
        cap = esso_capture.get(node) if esso_capture else None
        has_storage = (s_val > 0.0)
        soh_active, efc_cells = {}, {}
        if cap is not None:
            rated = cap.get('es_e_rated_per_unit') or {}
            soh = cap.get('es_soh_per_unit_cumul') or {}
            for key, r in rated.items():
                if r:
                    soh_active[key] = soh.get(key)
            for key, v in (cap.get('efc_per_day_per_cohort_year') or {}).items():
                if v:
                    efc_cells[key] = v.get('efc_per_day')
        floor_entries = per_node_floor.get(node, [])
        out[node] = {
            's_mva': s_val, 'e_mwh': e_val, 'has_storage': has_storage,
            'efc_per_day_max': cap.get('efc_per_day_max') if cap else None,
            'efc_per_day_per_cohort_year': efc_cells,
            'terminal_soh_per_active_cohort_year': soh_active,
            'terminal_soh_min_over_active_cohort_years': (
                min(v for v in soh_active.values() if v is not None)
                if any(v is not None for v in soh_active.values()) else None),
            'soh_floor_rows_active_at_terminal': sorted(
                f"({e.get('y_inv')}, {e.get('y')})" for e in floor_entries if e.get('active')),
            'published_available_capacity_terminal': (published_caps or {}).get(node),
            'note_zero_node': (None if has_storage else
                               'zero-capacity node: no active cohort; EFC undefined (None), SoH variables '
                               'fixed at 1.0 by production and not reported as a result'),
        }
    return out


def build_evaluation_record(*, spec, spec_path, spec_sha256, entry, report, component_levels,
                            floor_terminal, published_caps, peak_rss, wall, eval_dir, extra=None):
    """The STEP4_DFO_METHOD.md 2.5 record, built ONLY from the evaluation's own
    report/artifacts. Pure function (no I/O) -- exercised by the zero-solve
    checks on the D reference's committed artifacts."""
    import p515_s39_evaluate as E39  # generic helpers, BY IMPORT
    import p515_g_g1_g4_admm_gates as G

    rows = report.get('cycle_trajectory') or []
    cap = int(spec['cap'])
    required = int(spec['required_consecutive_cycles'])
    cert = E39._certification_from_trajectory(rows, cap, required) if rows else {'certified': False}
    stopped = G._derive_stopped_by_from_trajectory(rows, cap=cap, required_consecutive=required) if rows else None
    certified = bool(cert.get('certified'))
    last = rows[-1] if rows else {}
    bar = _max_step_last_n(rows)
    terminal_ratios = {c: {'primal': last.get(f'boyd_{c}_primal_ratio'),
                           'dual': last.get(f'boyd_{c}_dual_ratio'),
                           'max': E39._terminal_ratios(last)[c] if last else None} for c in CHANNELS}
    rc = (component_levels or {}).get('recourse_components') or {}
    status = 'certified' if certified else ('not_certified' if rows else 'no_trajectory')
    cause = None
    if not certified:
        cause = ('cap reached without certification' if rows and len(rows) >= cap else
                 'run ended before the cap without certification' if rows else 'no trajectory')
    record = {
        'schema': RECORD_SCHEMA,
        'campaign_id': spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256,
        'candidate_label': entry['label'],
        'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'],
        'instance_note': 'candidate_key = sha256 of the canonical candidate; identifies the problem instance',
        'working_dir_ids': entry['working_dir_ids'],
        'eval_dir': os.path.relpath(eval_dir, REPO),
        'configuration': spec['configuration'],
        'cap': cap,
        'required_consecutive_cycles': required,
        'status': status,
        'barrier': not certified,
        'barrier_cause': cause,
        'objective_convention': ('gross_operational_cost: settlement-EXCLUDED gross operational cost '
                                 '(the oracle cost convention); net_operational_recourse = gross minus '
                                 'terminal salvage credit'),
        'certified_cost': report.get('gross_operational_cost') if certified else None,
        'terminal_gross_operational_cost': report.get('gross_operational_cost'),
        'terminal_net_operational_recourse': rc.get('net_operational_recourse'),
        'bar': bar,
        'certification': cert,
        'certification_cycle': cert.get('certification_cycle'),
        'cycles_run': len(rows),
        'stopped_by_trajectory': stopped,
        'first_pass_cycle_per_channel': {c: E39._first_pass_cycle(rows, c) for c in CHANNELS},
        'terminal_ratios_per_channel': terminal_ratios,
        'rule_ten': {
            'reported_not_gated': True,
            'terminal_objective_change_abs': report.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': report.get('terminal_objective_tolerance'),
            'terminal_step_over_threshold': report.get('rule_ten_terminal_step_over_threshold'),
            'boyd_terminal_ratio_max_per_channel': {c: terminal_ratios[c]['max'] for c in CHANNELS},
        },
        'component_decomposition_totals_weighted': (component_levels or {}).get('totals_weighted'),
        'recourse_components': rc,
        'settlement_remainder': {
            'definition': ('recourse_components.interface_settlement_total: T_TSO + sum(T_DSO), the '
                           'non-cancelling interface settlement at the terminal point, excluded from the cost'),
            'value': rc.get('interface_settlement_total'),
        },
        'storage_per_node': _storage_per_node(entry['canonical'], report.get('esso_capture') or {},
                                              floor_terminal, published_caps),
        'local_solve_failures': report.get('local_solve_failures'),
        'network_failures_summary': report.get('network_failures_summary'),
        'solve_profile': report.get('solve_profile'),
        'wall_time_s': wall,
        'peak_rss': peak_rss,
    }
    if extra:
        record.update(extra)
    return record


# ==============================================================================
#  rule eleven for the record: capture paths asserted BEFORE the run
# ==============================================================================
RECORD_TRAJECTORY_FIELDS = tuple(sorted(set(PER_CYCLE_RECORD_FIELDS) | {
    'boyd_v_primal_ratio', 'boyd_pf_primal_ratio', 'boyd_ess_primal_ratio', 'boyd_v_dual_ratio',
    'boyd_pf_dual_ratio', 'boyd_ess_dual_ratio', 'boyd_v_channel_pass', 'boyd_pf_channel_pass',
    'boyd_ess_channel_pass', 'cycle_convergence', 'consecutive_converged_cycles',
    'objective_change_abs', 'objective_tolerance', 'gross_operational_cost', 'local_solves_ok'}))


def assert_record_capture_paths():
    """Fails fast (before any solve) if a quantity the 2.5 record needs has no capture path."""
    import inspect
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p514_n_instrumented_cstar as N
    import p515_s39_evaluate as E39
    src = inspect.getsource(srp)
    checks = {}
    derived_by_cycle_row = {'cycle', 'recourse', 'objective_change_ratio'}
    for field in RECORD_TRAJECTORY_FIELDS:
        checks[f'trajectory_field_{field}'] = (f"'{field}':" in src) or (field in derived_by_cycle_row)
    for name in ('es_avg_ch_dch_per_unit', 'es_soh_per_unit_cumul', 'es_e_rated_per_unit'):
        checks[f'esso_capture_attr_{name}'] = name in N.REQUIRED_ESSO_ATTRS
    for fn in ('run_admm_arm', 'write_boyd_terminal_s35ref', 's38_pf_capture_hooks',
               's39_exempt_until_capture_hooks', '_s35ref_terminal_floor_and_efc',
               '_derive_stopped_by_from_trajectory', '_identify_soh_floor_rows'):
        checks[f'harness_fn_{fn}'] = callable(getattr(G, fn, None))
    for fn in ('_certification_from_trajectory', '_first_pass_cycle', '_terminal_ratios'):
        checks[f'evaluator_fn_{fn}'] = callable(getattr(E39, fn, None))
    # settlement remainder: produced by production `_get_operational_recourse_components`
    # (key 'interface_settlement_total'), which `write_component_levels_terminal` calls.
    checks['component_levels_recourse_key_interface_settlement_total'] = (
        "'interface_settlement_total':" in inspect.getsource(srp._get_operational_recourse_components)
        and '_get_operational_recourse_components' in inspect.getsource(G.write_component_levels_terminal))
    checks['run_admm_arm_accepts_investment_map'] = (
        'investment_map' in inspect.signature(G.run_admm_arm).parameters)
    checks['run_admm_arm_post_run_hook_gets_state'] = ("'state' in inspect.signature(post_run_hook)"
                                                        in inspect.getsource(G.run_admm_arm))
    checks['production_state_has_peak_rss'] = "'peak_rss_ru_maxrss':" in src
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: capture paths missing for the evaluation record: {missing}')
    return checks


# ==============================================================================
#  the child (one evaluation)
# ==============================================================================
def _child_verify_env():
    bad = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f'CHILD REFUSES: thread caps not in force in the child environment: {bad}')
    return {k: os.environ.get(k) for k in THREAD_CAP_ENV}


def _child_stub(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started):
    """CHECKS ONLY (`p515_s44_campaign_harness_checks.py`): no model import, no
    solve -- sleeps, allocates, and writes a stub record through the same
    parent/child plumbing. Refused unless the frozen spec says test_only_stub."""
    if not spec.get('extra', {}).get('test_only_stub'):
        raise SystemExit('CHILD REFUSES: stub mode requested but the frozen spec is not a test-only stub spec')
    mode = args.stub_mode
    if mode == 'fail':
        print('stub child: failing on purpose', file=sys.stderr)
        raise SystemExit(3)
    blob = bytearray(int(args.stub_alloc_mb) * (1 << 20))
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    time.sleep(float(args.stub_sleep_s))
    record = {
        'schema': RECORD_SCHEMA, 'stub': True, 'campaign_id': spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': args.spec_sha256,
        'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'], 'status': 'stub', 'barrier': False,
        'child_pid': os.getpid(), 'child_ppid': os.getppid(), 'lock_content': lock_content,
        'thread_caps_seen': env_caps, 'started_t': started, 'ended_t': time.time(),
        'peak_rss': {'child_self_ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
    }
    _write_once_json(os.path.join(eval_dir, 'evaluation_record.json'), record)


def _config_hook_factory(spec, holder, overrides=None):
    """pre_solve_hook: verify the case file carries the D oracle configuration
    (same checks as `p515_s43_aa_run._aa_on_pre_solve_hook`), then apply the
    evaluation's overrides (`overrides`; default = the campaign-level
    `spec['configuration']['overrides']`; none for D). Only the AA flag and its
    reject-policy may be overridden (`validate_overrides`); after an AA override
    the frozen memory (5) and regularization (1e-10) are verified unchanged.
    Records into the report's rule_eleven_checklist (provenance)."""
    import p515_g_g1_g4_admm_gates as G
    if overrides is None:
        overrides = spec['configuration'].get('overrides') or {}
    overrides = validate_overrides(overrides)

    def hook(planning, sed, candidate, report):
        a = planning.params.admm
        checks = {
            'rho_v_matches_D': all(float(v) == G.S39_RHO_V for v in a.rho['v'].values()),
            'rho_pf_matches_D': all(float(v) == G.S39_RHO_PF for v in a.rho['pf'].values()),
            'rho_ess_matches_D': all(float(v) == G.S39_RHO_ESS for v in a.rho['ess'].values()),
            'tau_is_0': a.proximal_regularization['tso'].get('tau') == float(G.S39_TAU),
            'gamma_policy_tied_to_rho': a.proximal_regularization['tso'].get('gamma_policy') == 'tied_to_rho',
            'balancing_exempt_until_matches_D': (a.penalty_update.get('balancing_exempt_until')
                                                 == {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}),
            'balancing_exempt_channels_empty': not a.penalty_update.get('balancing_exempt_channels'),
            'freeze_after_unchanged_cycles_is_10': (a.penalty_update.get('freeze_after_unchanged_cycles')
                                                    == G.S39_FREEZE_AFTER_UNCHANGED_CYCLES),
            'freeze_backstop_cycle_is_200': a.penalty_update.get('freeze_backstop_cycle') == G.S39_FREEZE_BACKSTOP_CYCLE,
            'minimum_consecutive_converged_cycles_matches_spec': (
                a.minimum_consecutive_converged_cycles == int(spec['required_consecutive_cycles'])),
            'shared_ess_initialization_is_standalone': a.shared_ess_initialization == 'standalone',
            'num_max_iters_is_spec_cap': a.num_max_iters == int(spec['cap']),
            'anderson_acceleration_off_before_overrides': not a.anderson_acceleration.get('enabled'),
            'persistent_workers_off': not a.persistent_workers.get('enabled'),
            'parallel_execution_off': not planning.parallel_execution,
        }
        missing = sorted(k for k, v in checks.items() if not v)
        if missing:
            raise RuntimeError(f'S44 campaign child: configuration not as frozen (case file D + cap): {missing}')
        applied = {}
        for key, value in overrides.items():
            if key not in SUPPORTED_OVERRIDE_KEYS:
                raise RuntimeError(f'unsupported override {key}')
            if key == 'anderson_acceleration':
                merged = dict(a.anderson_acceleration)
                merged.update(value)
                a.anderson_acceleration = merged
                if a.anderson_acceleration != merged:
                    raise RuntimeError('anderson_acceleration override did not take effect')
                if (a.anderson_acceleration.get('memory') != FROZEN_AA_MEMORY
                        or a.anderson_acceleration.get('regularization') != FROZEN_AA_REGULARIZATION):
                    raise RuntimeError(f'anderson_acceleration memory/regularization not at the frozen '
                                       f'{FROZEN_AA_MEMORY}/{FROZEN_AA_REGULARIZATION}: {a.anderson_acceleration}')
                applied[key] = dict(a.anderson_acceleration)
        report.setdefault('rule_eleven_checklist', {})['s44_campaign_configuration_checks'] = checks
        report['rule_eleven_checklist']['s44_campaign_overrides_applied'] = applied
        holder['configuration_checks'] = checks
        holder['overrides_applied'] = applied
    return hook


# ==============================================================================
#  the optional POST-CERTIFICATION step (Addendum 25 item 2; P5_15_S44_GATE_RULING.md "Deviation")
# ==============================================================================
POST_CERTIFICATION_FILE = 'post_certification.json'
HULL_BOUND_DETAIL_FILE = 'hull_bound_detail.json'
AA_SIDECAR_FILE = 'aa_per_cycle.jsonl'


def assert_post_certification_capture_paths():
    """Rule eleven for the post-certification step: every function it reuses
    exists with the signature it is called with -- asserted in the child
    BEFORE the run whenever the evaluation requests the step."""
    import inspect
    import p515_s39_evaluate as E39
    import p515_g_g1_g4_admm_gates as G
    import p515_s41_hull_polish as HP
    import p515_s42_exact_fix_rerun as EF
    import p515_s43_aa_run as S43
    checks = {
        'certification_fn': callable(getattr(E39, '_certification_from_trajectory', None)),
        'stopped_by_fn': callable(getattr(G, '_derive_stopped_by_from_trajectory', None)),
        'hull_polish_fn': list(inspect.signature(HP._polish_all_blocks_hull).parameters)
        == ['planning', 'models', 'consensus_vars'],
        'persist_fn': list(inspect.signature(EF._persist_certified_models).parameters) == ['models', 'out_dir'],
        'decomposition_fn_accepts_reference_dir': 'reference_dir' in inspect.signature(
            S43._cost_decomposition_vs_d).parameters,
        'cost_band_constant_1_5e_4': S43.COST_RELATIVE_TOLERANCE == 1.5e-4,
        'reconciliation_tol_1_0': S43.RECONCILIATION_RESIDUAL_ABS_TOL == 1.0,
        'hull_gate_threshold_0_1_pct': HP.GATE_THRESHOLD_PCT == 0.1,
        'aa_sidecar_fn': callable(getattr(S43, '_build_aa_per_cycle_sidecar', None)),
        'run_admm_arm_passes_state_to_hook': ("'state' in inspect.signature(post_run_hook)"
                                              in inspect.getsource(G.run_admm_arm)),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: post-certification capture paths missing: {missing}')
    return checks


def non_degenerate_hull_counts(hull_bound_detail):
    """Per channel: hull descriptors and ACTIVE descriptors EXCLUDING degenerate
    intervals (Addendum 24 convention, as `p515_s42_hull_counts.py` applied to
    the committed Step 3.5 evidence; `_hull_bounds_active` counts a degenerate
    interval as active by definition)."""
    out = {}
    for d in hull_bound_detail or []:
        c = out.setdefault(d['channel'], {'total': 0, 'degenerate': 0, 'non_degenerate': 0,
                                          'active_non_degenerate': 0})
        c['total'] += 1
        if d['degenerate']:
            c['degenerate'] += 1
        else:
            c['non_degenerate'] += 1
            if d['active']:
                c['active_non_degenerate'] += 1
    return out


def aa_sidecar_summary(rows):
    """Counts over the AA per-cycle fields already in the trajectory."""
    actions = {}
    retained = []
    for r in rows:
        act = r.get('aa_action')
        actions[act] = actions.get(act, 0) + 1
        if isinstance(act, str) and act.startswith('rejected') and (r.get('aa_memory_size_after') or 0) > 0:
            retained.append({'cycle': r.get('cycle'), 'memory_size_before': r.get('aa_memory_size_before'),
                             'memory_size_after': r.get('aa_memory_size_after'),
                             'rho_changed_channels': r.get('aa_rho_changed_channels')})
    return {'n_rows': len(rows), 'action_counts': actions,
            'n_accepted': sum(1 for r in rows if r.get('aa_accepted') is True),
            'n_rejected': sum(v for k, v in actions.items() if isinstance(k, str) and k.startswith('rejected')),
            'rejections_with_memory_retained': retained,
            'first_accept_cycle': next((r.get('cycle') for r in rows if r.get('aa_accepted') is True), None),
            'first_reject_cycle': next((r.get('cycle') for r in rows if isinstance(r.get('aa_action'), str)
                                        and r['aa_action'].startswith('rejected')), None)}


def run_post_certification(*, planning, models, rows, report, state, spec, entry, eval_dir,
                           polish_fn=None, persist_fn=None, decomposition_fn=None):
    """The optional post-certification step, IN the child, inside `run_admm_arm`'s
    post_run_hook (same live models / state, after `write_boyd_terminal_s35ref`
    wrote component_levels_terminal.json). Order as `p515_s43_aa_run.py`:
    certification test -> (b) cost vs reference -> (c) decomposition vs
    reference -> persist certified models (BEFORE the polish mutates them) ->
    (d) interval-hull polish. Skipped cleanly, with the reason recorded, when
    the trajectory is not certified under the spec's own bar. The *_fn
    parameters exist only so the zero-solve checks can substitute fakes for the
    two solving/pickling calls; production callers pass nothing."""
    import p515_s39_evaluate as E39
    import p515_g_g1_g4_admm_gates as G
    import p515_s41_hull_polish as HP
    import p515_s42_exact_fix_rerun as EF
    import p515_s43_aa_run as S43
    polish_fn = polish_fn or HP._polish_all_blocks_hull
    persist_fn = persist_fn or EF._persist_certified_models
    decomposition_fn = decomposition_fn or S43._cost_decomposition_vs_d

    request = entry.get('post_certification')
    cap, required = int(spec['cap']), int(spec['required_consecutive_cycles'])
    out = {'requested': request, 'status': None}
    if not request:
        out.update(status='not_requested', evaluated=False)
        return out, None
    if not rows:
        out.update(status='skipped', evaluated=False, skip_reason='no trajectory')
        return out, None
    cert = E39._certification_from_trajectory(rows, cap, required)
    stopped = G._derive_stopped_by_from_trajectory(rows, cap=cap, required_consecutive=required)
    out['certification'] = cert
    out['stopped_by'] = stopped
    if not cert.get('certified'):
        out.update(status='skipped', evaluated=False, skip_reason=(
            f"trajectory not certified under the spec's bar (cycles_run={cert.get('cycles_run')}, cap={cap}, "
            f"required_consecutive={required}, terminal_consecutive_converged_cycles="
            f"{cert.get('terminal_consecutive_converged_cycles')}, stopped_by={stopped.get('stopped_by')!r}); "
            'post-certification items are never evaluated at an uncertified point'))
        return out, None
    out['evaluated'] = True
    q = report.get('gross_operational_cost')
    out['certified_cost'] = q
    out['objective_convention'] = 'gross_operational_cost (settlement-excluded), as the evaluation record'

    ref = request.get('reference')
    if ref:
        verify_reference_unchanged(ref)
        q_ref = ref['certified_cost']
        abs_tol = S43.COST_RELATIVE_TOLERANCE * q_ref
        abs_diff = abs(q - q_ref) if q is not None else None
        out['gate_b_cost_vs_reference'] = {
            'definition': '|Q - Q_ref| <= 1.5e-4 * Q_ref (p515_s43_aa_run gate (b), reference = the D evaluation '
                          'of the same candidate)',
            'certified_cost': q, 'reference_certified_cost': q_ref, 'reference_eval_dir': ref['eval_dir'],
            'abs_diff': abs_diff, 'relative_tolerance': S43.COST_RELATIVE_TOLERANCE, 'abs_tolerance': abs_tol,
            'pass': bool(abs_diff is not None and abs_diff <= abs_tol)}
        with open(os.path.join(eval_dir, 'component_levels_terminal.json')) as handle:
            my_cl = json.load(handle)
        decomposition = decomposition_fn(my_cl, q, reference_dir=os.path.join(REPO, ref['eval_dir']))
        decomposition['labels_note'] = ("keys named 'D'/'AA' by p515_s43_aa_run: 'D' = the reference evaluation, "
                                        "'AA' = this evaluation")
        out['gate_c_cost_decomposition_vs_reference'] = decomposition
        out['gate_c_pass'] = bool(decomposition.get('reconciles'))
    else:
        out['gate_b_cost_vs_reference'] = None
        out['gate_c_cost_decomposition_vs_reference'] = None
        out['gate_c_pass'] = None

    if request.get('persist_certified_models'):
        out['persisted_models'] = persist_fn(models, eval_dir)
    else:
        out['persisted_models'] = None

    hull_bound_detail = None
    if request.get('hull_polish'):
        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('post-certification hull polish: state/consensus_vars not available')
        t0 = time.time()
        polish, hull_bound_detail = polish_fn(planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - t0
        gate = polish.get('gate')
        n_solved = sum(1 for b in polish.get('per_block') or [] if b.get('solved'))
        out['gate_d_hull_polish'] = {
            'definition': ('p515_s41_hull_polish gate: |sum over blocks of [f_i(polished) - f_i(certified)]| / '
                           'certified cost < 0.1 %, evaluated only when every block solves'),
            'blocks_solved': n_solved, 'n_blocks': polish.get('n_blocks'), 'all_solved': polish.get('all_solved'),
            'failed_blocks': polish.get('failed_blocks'),
            'delta_sum_blocks': gate.get('delta_sum_blocks') if gate else None,
            'relative_pct': gate.get('relative_pct') if gate else None,
            'threshold_pct': HP.GATE_THRESHOLD_PCT,
            'settlement_excluded_change': (gate['reported_not_gated']['gross_operational_cost_change_settlement_excluded']
                                           if gate else None),
            'settlement_remainder_before': (gate['reported_not_gated']['interface_settlement_total_before']
                                            if gate else None),
            'settlement_remainder_after': (gate['reported_not_gated']['interface_settlement_total_after']
                                           if gate else None),
            'hull_bounds_active_by_channel_incl_degenerate': polish.get('hull_bounds_active_by_channel'),
            'hull_bounds_non_degenerate_by_channel': non_degenerate_hull_counts(hull_bound_detail),
            'flagged_blocks': polish.get('flagged_blocks'), 'flag_abs_threshold': polish.get('flag_abs_threshold'),
            'solve_profile': polish.get('solve_profile'), 'runtime_s': polish['runtime_s'],
            'pass': bool(gate is not None and polish.get('all_solved') and gate.get('pass')),
        }
        out['hull_polish_full'] = polish
        path = os.path.join(eval_dir, HULL_BOUND_DETAIL_FILE)
        _write_once_json(path, hull_bound_detail)
        out['hull_bound_detail_path'] = os.path.relpath(path, REPO)
    else:
        out['gate_d_hull_polish'] = None
    out['status'] = 'evaluated'
    return out, hull_bound_detail


def post_certification_summary(pc):
    """The compact block copied into evaluation_record.json (full detail stays in post_certification.json)."""
    if pc is None:
        return None
    s = {k: pc.get(k) for k in ('status', 'evaluated', 'skip_reason', 'error', 'certified_cost',
                                 'gate_c_pass', 'persisted_models', 'hull_bound_detail_path')}
    s['requested'] = pc.get('requested')
    b = pc.get('gate_b_cost_vs_reference')
    s['gate_b'] = ({k: b.get(k) for k in ('abs_diff', 'abs_tolerance', 'reference_certified_cost', 'pass')}
                   if b else None)
    c = pc.get('gate_c_cost_decomposition_vs_reference')
    s['gate_c'] = ({k: c.get(k) for k in ('headline_diff_AA_minus_D', 'dominant_two_diff', 'unaccounted_residual',
                                          'other_priced_components_nonzero', 'reconciles')} if c else None)
    d = pc.get('gate_d_hull_polish')
    s['gate_d'] = ({k: d.get(k) for k in ('blocks_solved', 'n_blocks', 'relative_pct', 'threshold_pct',
                                          'settlement_excluded_change', 'settlement_remainder_before',
                                          'settlement_remainder_after', 'hull_bounds_non_degenerate_by_channel',
                                          'pass')} if d else None)
    return s


def _child_real(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started):
    import pyomo.environ as pe  # noqa: F401
    import p515_g_g1_g4_admm_gates as G
    from p515_s40_polish_gap import _build_floor_rows

    capture_checklist = assert_record_capture_paths()
    eff_overrides = validate_overrides(entry['overrides'] if 'overrides' in entry
                                       else (spec['configuration'].get('overrides') or {}))
    aa_on = bool((eff_overrides.get('anderson_acceleration') or {}).get('enabled'))
    post_request = entry.get('post_certification')
    post_checklist = None
    if post_request:
        post_checklist = assert_post_certification_capture_paths()
        if post_request.get('reference'):
            post_checklist['reference_hashes_verified_before_run'] = verify_reference_unchanged(
                post_request['reference'])
    ids = entry['working_dir_ids']
    for eid in ids.values():
        if os.path.exists(os.path.join(G.O.WORK_DIR, eid)):
            raise RuntimeError(f'working dir id already used (never reusable): {eid}')
    label = spec['configuration']['arm_label']
    investment_map = investment_map_from_canonical(entry['canonical'])
    if entry['canonical']['investment_year'] != G.N.INVEST_YEAR:
        raise RuntimeError(f"candidate investment year {entry['canonical']['investment_year']} != "
                           f'run_admm_arm investment year {G.N.INVEST_YEAR}')

    _cc, floor_rows_by_node, _fc = _build_floor_rows(ids['precheck'])
    paths = {
        'recourse_jump': os.path.join(eval_dir, 'recourse_jump_sidecar_baseline.jsonl'),
        'ess_stride': os.path.join(eval_dir, 'ess_entry_stride_baseline.jsonl'),
        'floor': os.path.join(eval_dir, 'soh_floor_sidecar_baseline.jsonl'),
        'pf_stride': os.path.join(eval_dir, f'pf_entry_stride_{label}.jsonl'),
        'exempt': os.path.join(eval_dir, f'ess_exempt_until_state_{label}.jsonl'),
    }
    for p in list(paths.values()) + [os.path.join(eval_dir, f) for f in (
            POST_CERTIFICATION_FILE, HULL_BOUND_DETAIL_FILE, AA_SIDECAR_FILE, 'certified_models.pkl')]:
        if os.path.exists(p):
            raise RuntimeError(f'refusing to overwrite existing artifact: {p}')

    holder = {}

    def post_run_hook(planning, sed, models, rows, report, out_dir, label, state=None):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(paths['recourse_jump'], REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(paths['ess_stride'], REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(paths['floor'], REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(paths['pf_stride'], REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(paths['exempt'], REPO)
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node, floor_sidecar_path=paths['floor'])
        caps = sed.get_updated_capacities(models['esso'])
        holder['published_caps'] = {str(n): {str(y): v for y, v in per_year.items()} for n, per_year in caps.items()}
        st = state or {}
        holder['peak_rss_ru_maxrss_production'] = st.get('peak_rss_ru_maxrss')
        holder['peak_rss_platform_units'] = st.get('peak_rss_platform_units')
        if aa_on:
            import p515_s43_aa_run as S43  # its sidecar builder, BY IMPORT, unchanged
            S43._build_aa_per_cycle_sidecar(rows, os.path.join(eval_dir, AA_SIDECAR_FILE))
            holder['aa_sidecar'] = {'path': os.path.relpath(os.path.join(eval_dir, AA_SIDECAR_FILE), REPO),
                                    **aa_sidecar_summary(rows)}
        if post_request:
            t_pc = time.time()
            try:
                pc, _detail = run_post_certification(planning=planning, models=models, rows=rows, report=report,
                                                     state=state, spec=spec, entry=entry, eval_dir=eval_dir)
            except Exception as error:  # noqa: BLE001 -- recorded loudly; the evaluation itself stands
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                pc = {'requested': post_request, 'status': 'error', 'evaluated': False,
                      'error': f'{type(error).__name__}: {error}', 'traceback': tb}
            pc['runtime_s'] = time.time() - t_pc
            pc['production_peak_rss_before_step'] = st.get('peak_rss_ru_maxrss')
            pc['process_ru_maxrss_after_step'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _write_once_json(os.path.join(eval_dir, POST_CERTIFICATION_FILE), pc)
            holder['post_certification'] = pc
            print(f"[S44-CHILD] post-certification: status={pc.get('status')} "
                  f"reason={pc.get('skip_reason') or pc.get('error')}", flush=True)

    t0 = time.time()
    with G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                paths['pf_stride'], floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(paths['exempt']):
        report, report_path = G.run_admm_arm(
            label, eval_dir, k_override=None, investment_map=investment_map,
            num_max_iters_override=int(spec['cap']), eval_id=ids['run'], apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=post_run_hook,
            pre_solve_hook=_config_hook_factory(spec, holder, overrides=eff_overrides))
    run_wall = time.time() - t0

    rows = report.get('cycle_trajectory') or []
    per_cycle_path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    if os.path.exists(per_cycle_path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {per_cycle_path}')
    with open(per_cycle_path, 'w') as handle:
        for r in rows:
            handle.write(json.dumps({k: r.get(k) for k in PER_CYCLE_RECORD_FIELDS}, default=str) + '\n')

    with open(os.path.join(eval_dir, 'component_levels_terminal.json')) as handle:
        component_levels = json.load(handle)
    with open(os.path.join(eval_dir, 'boyd_terminal.json')) as handle:
        boyd_terminal = json.load(handle)
    self_ru = resource.getrusage(resource.RUSAGE_SELF)
    children_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = {
        'units': 'bytes on macOS/BSD, kilobytes on Linux (ru_maxrss)',
        'child_python_process_ru_maxrss': self_ru.ru_maxrss,
        'production_state_peak_rss_ru_maxrss': holder.get('peak_rss_ru_maxrss_production'),
        'solver_subprocesses_max_ru_maxrss': children_ru.ru_maxrss,
        'semantics': ('child_python_process = RUSAGE_SELF of the evaluation process at record time '
                      '(the evaluation\'s own peak); production_state = the same measure taken by '
                      'production at the end of run_operational_planning; solver_subprocesses = '
                      'RUSAGE_CHILDREN max over the IPOPT executables this evaluation launched; when a '
                      'post-certification step ran, child_python_process includes it (the ADMM run alone is '
                      'production_state)'),
    }
    wall = {'child_process_s': time.time() - started, 'run_admm_arm_s': run_wall,
            'run_admm_arm_reported_wall_clock_s': report.get('wall_clock_s')}
    record = build_evaluation_record(
        spec=spec, spec_path=spec_path, spec_sha256=args.spec_sha256, entry=entry, report=report,
        component_levels=component_levels,
        floor_terminal=boyd_terminal.get('soh_floor_multiplier_and_efc_per_cohort_year_terminal'),
        published_caps=holder.get('published_caps'), peak_rss=peak_rss, wall=wall, eval_dir=eval_dir,
        extra={
            'record_capture_checklist_asserted_before_run': capture_checklist,
            'eval_key': _entry_eval_key(entry),
            'evaluation_overrides_effective': eff_overrides,
            'post_certification_capture_checklist_asserted_before_run': post_checklist,
            'post_certification': post_certification_summary(holder.get('post_certification')),
            'post_certification_path': (os.path.relpath(os.path.join(eval_dir, POST_CERTIFICATION_FILE), REPO)
                                        if holder.get('post_certification') is not None else None),
            'aa_per_cycle': holder.get('aa_sidecar'),
            'configuration_checks_in_child': holder.get('configuration_checks'),
            'overrides_applied_in_child': holder.get('overrides_applied'),
            'thread_caps_seen_by_child': env_caps,
            'PYTHONHASHSEED_in_child': os.environ.get('PYTHONHASHSEED'),
            'nlp_solver_path_in_child': os.environ.get('NLP_SOLVER_PATH'),
            'campaign_lock_seen_by_child': lock_content,
            'child_pid': os.getpid(), 'parent_pid': os.getppid(),
            'report_path': os.path.relpath(report_path, REPO),
            'per_cycle_record_path': os.path.relpath(per_cycle_path, REPO),
        })
    _write_once_json(os.path.join(eval_dir, 'evaluation_record.json'), record)
    manifest = {}
    for root, _dirs, files in os.walk(eval_dir):
        for fname in sorted(files):
            if fname in ('child_stdout.log', 'child_stderr.log', 'exit_code.txt', 'wait4_rusage.json'):
                continue  # still being written by the parent; the campaign manifest hashes them
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = sha256_file(fpath)
    _write_once_json(os.path.join(eval_dir, 'child_manifest_sha256.json'), manifest)
    print(f"[S44-CHILD] {entry['label']}: status={record['status']} cycles={record['cycles_run']} "
          f"certified_cost={record['certified_cost']} bar={record['bar']['value']} "
          f"peak_rss={self_ru.ru_maxrss}")
    return {'post_certification_error': (holder.get('post_certification') or {}).get('status') == 'error'}


def main_child(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true', required=True)
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--eval-key', required=True)
    parser.add_argument('--stub-mode', default=None, choices=(None, 'ok', 'fail'))
    parser.add_argument('--stub-sleep-s', default='2')
    parser.add_argument('--stub-alloc-mb', default='64')
    parser.add_argument('--lock-path', default=CAMPAIGN_LOCK_PATH)
    args = parser.parse_args(argv)
    started = time.time()
    env_caps = _child_verify_env()
    spec_path, spec = load_frozen_spec(args.campaign_root, args.spec_sha256)
    lock_content = verify_child_lock(args.spec_sha256, lock_path=args.lock_path)
    entry = next((e for e in spec['candidates'] if _entry_eval_key(e) == args.eval_key), None)
    if entry is None:
        raise SystemExit(f'CHILD REFUSES: eval key {args.eval_key} not in the frozen spec')
    eval_dir = os.path.join(args.campaign_root, 'evals', entry['eval_dir'])
    if not os.path.isdir(eval_dir) or os.path.exists(os.path.join(eval_dir, 'evaluation_record.json')):
        raise SystemExit(f'CHILD REFUSES: eval dir missing or already holds a record: {eval_dir}')
    print(f"[S44-CHILD] pid={os.getpid()} ppid={os.getppid()} label={entry['label']} key={entry['key'][:16]} "
          f"caps={env_caps} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}", flush=True)
    try:
        if args.stub_mode is not None:
            _child_stub(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started)
        else:
            outcome = _child_real(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started)
            if outcome and outcome.get('post_certification_error'):
                print('[S44-CHILD] post-certification step FAILED (recorded in post_certification.json); '
                      'exiting 2', file=sys.stderr, flush=True)
                sys.exit(2)
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 -- recorded as a barrier with its cause, then exit 1
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        record_path = os.path.join(eval_dir, 'evaluation_record.json')
        if not os.path.exists(record_path):
            _write_once_json(record_path, {
                'schema': RECORD_SCHEMA, 'campaign_id': spec['campaign_id'],
                'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': args.spec_sha256,
                'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
                'candidate_key': entry['key'], 'status': 'error', 'barrier': True,
                'barrier_cause': f'{type(error).__name__}: {error}', 'traceback': tb,
                'wall_time_s': {'child_process_s': time.time() - started},
            })
        sys.exit(1)


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        main_child(sys.argv[1:])
    else:
        raise SystemExit('p515_s44_campaign_harness.py is a library + child entry point; '
                         'campaigns are launched by their own script (e.g. p515_s44_gate.py)')
