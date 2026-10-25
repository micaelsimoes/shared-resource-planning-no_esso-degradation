"""
P5.15 Addendum 83 (W183 order; task W183-0, reusable by W183c / W183d) -- ZERO-SOLVE NL + component-digest
reference of the model files production would hand IPOPT at the ADMM initialisation, for a grid of configurations,
built through the configuration path real campaign children use. Model BUILDS are permitted (Addendum 83 names
them); nothing is solved.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 83, ruling 3 (iv) (the NRF-off NL identity gate needs a pre-change
reference) and (vii) (gate G0: at x = 0 the ESSO and network NL files with `ageing_enabled` on and off are
byte-identical; `model_variant` + `flex_price_multiplier` apply together); the W183 order (W183c, W183d).

WHAT IS BUILT (per configuration; each builder in its OWN freshly constructed planning object)
  * The planning object is built by `p515_g_g1_g4_admm_gates._construct_arm_planning` (the code `run_admm_arm`
    calls; arm 's39_D', case-file rho, cap CAP, cohort year 2025) and passed through the campaign child's
    configuration hook `p515_s44_campaign_harness._config_hook_factory` exactly as the child calls it (spec-like dict
    = the ext-v6 configuration block: case-file AA declared, ESS ageing baseline declared and pinned to the file's
    sha256, convergence-depth tail declared enabled; `model_variant` and `flex_price_multiplier` as the entry
    options; `expected_floor_rows` from `p515_s40_polish_gap._build_floor_rows`, as the child does).
  * SEQUENTIAL builder (the production path every campaign child runs: `parallel_execution` is False in the case
    file and the hook refuses otherwise): `planning.run_operational_planning(type='distributed', ...)` itself, with
    `optimize` replaced ON THE INSTANCE (TSO, every DSO, the ESSO) by a stub that records the blocks it is handed and
    returns None per block; production reads that as a failed initialisation and returns before
    `_prepare_*` / `update_*_to_admm` (shared_resources_planning.py `_run_operational_planning`, the
    `initialization_failed` branch). Captured: TSO 12 blocks, DSO 3 x 12 blocks, ESSO 3 node models.
  * PARALLEL builder: `shared_resources_planning.create_distribution_network_model(node_id, dn,
    candidate['total_capacity'], alpha, floor)` -- the ProcessPool worker function -- called IN-PROCESS for every
    DSO with the same stub (precedent `p515_s53_init_fix_zero_solve_checks.py` A2). Captured: DSO 3 x 12 blocks.
    The two DSO builders name their interface rows differently; each is compared only with itself.
  * Per block, at the moment the stub is called (= what IPOPT would receive):
      - component digest: `block_state` below, VERBATIM from `p515_s47_case_file_baseline_check.py` (asserted
        before the run by comparing the two function sources); per-index items written to a gz JSON per block;
        plus the `active` flag of every constraint index (for the "every differing index is inactive" question);
      - NL file written by `block.write(path, format='nl', io_options={'symbolic_solver_labels': False})` (the
        production form: `NLWriter.__call__` forces linear_presolve and scale_model off, the path
        `SolverFactory('ipopt').solve` takes) and a second copy with `symbolic_solver_labels=True` (+ .row/.col).
        sha256 of every file; the labelled copy with comments stripped is checked equal to the plain copy with
        comments stripped, except the header's 'max name lengths' line (0 0 without labels) -- so the .row/.col
        names index the plain file. Files are then gzip'ed (mtime 0).
        The digest is taken BEFORE the NL writes (`Block.write` registers a symbol map on the block).

GRID (default): {x0 = 0 everywhere, unit = node 7 0.25 MVA / 1.0 MWh 2025} x {A = baseline (no model_variant),
N = no ageing (model_variant ageing_enabled False, soh point 'end', calendar retention 1.0, eol_retention_r 0.50 --
the T8 / ext v3 `e_no_ageing` declaration)} x {flex_price_multiplier 1.0 (explicit; runs the override path and its
read-back; key-neutral), 2.0} -> 8 configurations. `--configs` restricts it.

COMPARISONS (only those whose configurations were built): (i) x0 A vs N at 1x and 2x; (ii) unit A vs N at 2x
(and 1x), with the .row/.col per-row analysis of every ESSO NL that differs; (iii) 1x vs 2x within each
(instance, model); (iv) DETERMINISM: unless `--no-determinism`, the parent re-builds `--determinism-configs` in a
SEPARATE process (this script, `--no-determinism`, output `<out>/determinism_child/`) and compares every NL sha256
(plain, labelled, .row, .col) and every component digest.

Nothing here solves: `SolveProfileGuard(permitted=())` is installed before any project import and verified at
exactly 0 at the end (and in the child). Stub call counts are checked exactly (TSO 1, each DSO 1, ESSO 1 per
sequential build; each DSO 1 per parallel build) -- too few means the path did not run.

PROVENANCE: git HEAD; sha256 + git status of every production module (tracked root-level *.py not matching p\\d)
and of every repo module imported (at start and at end); `--expect-production-commit C` refuses unless every
production module's working-tree content equals its blob at C; the pyomo version; the interpreter; the
case / ESS params / cost files' sha256.

OUTPUT (write-once; the directory may pre-exist containing only `launch.log`):
  <out>/w183_nl_digest.json          the result (gate_result_io)
  <out>/nl/<config>/<builder>/*.gz    NL files (plain .nl, labelled .sym.nl, .sym.row, .sym.col), gzip mtime 0
  <out>/items/<config>/<builder>/*.json.gz   per-block per-index component items
  <out>/manifest_sha256.json          sha256 of every file written (raw-content sha256 of NL files is in the JSON)
  <out>/determinism_child/...         the child's own output (same layout)

LAUNCH (attached, alone, both streams captured; the harness committed first):
  mkdir -p data/SRP1/Results/P515S55/w183_nl_reference_pre
  NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w183_nl_digest.py --out data/SRP1/Results/P515S55/w183_nl_reference_pre --label pre \\
      --scratch <scratchpad>/w183_0_build --expect-production-commit 575bfec4 \\
      > data/SRP1/Results/P515S55/w183_nl_reference_pre/launch.log 2>&1
Exit 0 when every check passes (the comparisons are REPORTED, not gated; the checks are the harness's own: guard 0,
stub counts, read-backs, determinism, the stripped-label equality), 1 otherwise.
"""

import argparse
import ast
import gc
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183 NL digest (zero solves, builds only)').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
STAGE = ('P5.15 Addendum 83 W183-0 / W183c / W183d -- NL + component-digest reference of the ADMM initialisation '
         'models (zero solves, builds only)')
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (iv), (vii); W183 order',
             'P5_15_W182E_CODE_PATHS.md (Z1, Not confirmed)']
PRECEDENT_REL = 'p515_s47_case_file_baseline_check.py'
CAP = 300
REQUIRED_CONSECUTIVE = 10
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CONVERGENCE_DEPTH_TAIL = {'compl_inf_tol': 1e-06, 'enabled': True}
INVESTMENT_YEAR = 2025
INSTANCES = {'x0': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)},
             'unit': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}}
NO_AGEING = {'ageing_enabled': False, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0,
             'eol_retention_r': 0.5}
MODELS = {'A': None, 'N': NO_AGEING}
FLEX = {'1x': 1.0, '2x': 2.0}
ALL_CONFIGS = [f'{i}_{m}_{f}' for i in INSTANCES for m in MODELS for f in FLEX]
DEFAULT_DETERMINISM = 'x0_A_1x,unit_N_2x'
# The committed ext-v6 e_no_ageing spec's pre-launch record: the base key (no resettle) of the unit under the no-ageing
# variant in the current configuration (campaign_spec_s53_w142_resettle_ext_v6_e_no_ageing_10f4aa7a.json,
# extra.pre_launch_assertion_at_freeze.base_key_current_configuration). Cross-check of the key computation only.
EXT_V6_UNIT_N_BASE_KEY = '054976fb6f0c6adbd3175a23f60915e931e161f71e3bd59c8719e71d3ffa000a'
DEGRADATION_ROW_RE = re.compile(r'^energy_storage_capacity_degradation\[\d+\]$')
NL_SEGMENT_HEADS = set('CJOVFSLGdxrbk')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183-nl] {msg}', flush=True)


def _git(args, check=True):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=check).stdout


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _safe(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text))


# ======================================================================================================================
#  digests -- `block_state` is VERBATIM from p515_s47_case_file_baseline_check.py (asserted in `assert_capture_paths`)
# ======================================================================================================================
def block_state(model, keep_items):
    """Per component: {index repr: state repr}. Returns (component digests, per-index items or None)."""
    import pyomo.environ as pe
    comps = {}
    for comp in model.component_objects(pe.Param, active=None, descend_into=True):
        comps[f'Param:{comp.name}'] = {repr(i): repr(pe.value(comp[i], exception=False)) for i in comp}
    for comp in model.component_objects(pe.Var, active=None, descend_into=True):
        comps[f'Var:{comp.name}'] = {repr(i): repr((v.value, v._lb, v._ub, v.fixed)) for i, v in comp.items()}
    for comp in model.component_objects(pe.Constraint, active=None, descend_into=True):
        comps[f'Constraint:{comp.name}'] = {
            repr(i): repr((c.active, None if c.lower is None else pe.value(c.lower),
                           None if c.upper is None else pe.value(c.upper), str(c.body))) for i, c in comp.items()}
    for comp in model.component_objects(pe.Objective, active=None, descend_into=True):
        comps[f'Objective:{comp.name}'] = {repr(i): repr((o.active, str(o.expr))) for i, o in comp.items()}
    for comp in model.component_objects(pe.Expression, active=None, descend_into=True):
        comps[f'Expression:{comp.name}'] = {repr(i): str(e.expr) for i, e in comp.items()}
    comps['Blocks:active'] = {blk.name or '<root>': repr(blk.active)
                              for blk in model.block_data_objects(active=None, descend_into=True)}
    comp_digests = {name: _h(json.dumps(items, sort_keys=True)) for name, items in comps.items()}
    block = _h(json.dumps(comp_digests, sort_keys=True))
    return block, comp_digests, (comps if keep_items else None)


def constraint_active_flags(model):
    import pyomo.environ as pe
    return {f'Constraint:{comp.name}': {repr(i): bool(c.active) for i, c in comp.items()}
            for comp in model.component_objects(pe.Constraint, active=None, descend_into=True)}


def _str_keys(obj):
    """JSON object keys must be strings: a non-string key (e.g. the tuple keys of production's floor-row map) is
    written as its repr. Values are untouched."""
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else repr(k)): _str_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_str_keys(v) for v in obj]
    return obj


def _items_for_json(items):
    """Per-index items as written to the gz item file: the `Blocks:active` reprs become JSON booleans (gate_result_io
    refuses a string flag); every other state is its repr / str exactly as digested."""
    out = {}
    for name, entries in items.items():
        if name == 'Blocks:active':
            out[name] = {k: (v == 'True') for k, v in entries.items()}
        else:
            out[name] = entries
    return out


def _gzip_write_bytes(path, data):
    with open(path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as gz:
            gz.write(data)


def _gzip_file(src, dst):
    with open(src, 'rb') as handle:
        _gzip_write_bytes(dst, handle.read())
    os.remove(src)


def _strip_nl_comments(text):
    return '\n'.join(line.split('#', 1)[0].rstrip() for line in text.splitlines())


def _drop_name_length_line(text):
    lines = text.splitlines()
    return '\n'.join(line for i, line in enumerate(lines) if not (i < 10 and 'max name lengths' in line))


def write_nl(model, nl_dir, stem):
    """Write the block's NL twice (labels off = production form; labels on, + .row/.col), hash the raw files, check the
    two equal after comment stripping, gzip them (mtime 0). Returns the record; never raises (errors are recorded)."""
    rec = {'plain': None, 'symbolic': None, 'stripped_equal': None, 'error': None}
    plain = os.path.join(nl_dir, f'{stem}.nl')
    sym = os.path.join(nl_dir, f'{stem}.sym.nl')
    try:
        model.write(plain, format='nl', io_options={'symbolic_solver_labels': False})
        model.write(sym, format='nl', io_options={'symbolic_solver_labels': True})
    except Exception as error:  # noqa: BLE001 -- recorded, the check `no_nl_write_errors` fails
        rec['error'] = f'{type(error).__name__}: {error}'
        return rec
    with open(plain) as handle:
        plain_text = handle.read()
    with open(sym) as handle:
        sym_text = handle.read()
    # The one header line that legitimately differs: 'max name lengths' (0 0 without labels).
    rec['stripped_equal'] = (_strip_nl_comments(_drop_name_length_line(plain_text))
                             == _strip_nl_comments(_drop_name_length_line(sym_text)))
    files = {'plain': plain, 'symbolic': sym, 'row': os.path.join(nl_dir, f'{stem}.sym.row'),
             'col': os.path.join(nl_dir, f'{stem}.sym.col')}
    out = {}
    for name, path in files.items():
        out[name] = {'sha256': _sha_file(path), 'bytes': os.path.getsize(path),
                     'gz': os.path.relpath(path + '.gz', REPO)}
        _gzip_file(path, path + '.gz')
    rec['plain'] = out['plain']
    rec['symbolic'] = {'nl': out['symbolic'], 'row': out['row'], 'col': out['col']}
    return rec


class Capture:
    """The recording stub state for one build: blocks digested and NL-written at the would-be solve."""

    def __init__(self, out, config, builder):
        self.config = config
        self.builder = builder
        self.nl_dir = os.path.join(out, 'nl', config, builder)
        self.items_dir = os.path.join(out, 'items', config, builder)
        os.makedirs(self.nl_dir)
        os.makedirs(self.items_dir)
        self.blocks = {}
        self.calls = {}

    def block(self, key, model):
        digest, comps, items = block_state(model, keep_items=True)
        active = constraint_active_flags(model)
        items_path = os.path.join(self.items_dir, f'{_safe(key)}.json.gz')
        payload = GRIO.dumps({'block': key, 'items': _items_for_json(items), 'constraint_active': active},
                             sort_keys=True).encode()
        _gzip_write_bytes(items_path, payload)
        del items
        nl = write_nl(model, self.nl_dir, _safe(key))
        self.blocks[key] = {'sha256': digest, 'components': comps,
                            'items_gz': os.path.relpath(items_path, REPO), 'items_gz_sha256': _sha_file(items_path),
                            'nl': nl}

    def network_stub(self, tag):
        def stub(model, *args, **kwargs):
            self.calls[tag] = self.calls.get(tag, 0) + 1
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    self.block(f'{tag}|{year}|{day}', model[year][day])
                    results[year][day] = None
            return results
        return stub

    def esso_stub(self, models, *args, **kwargs):
        self.calls['ESSO'] = self.calls.get('ESSO', 0) + 1
        for node_id, model in models.items():
            self.block(f'ESSO|node{node_id}', model)
        return {node_id: None for node_id in models}


# ======================================================================================================================
#  configuration (the campaign child's path)
# ======================================================================================================================
def spec_like(declared_ageing, pin):
    """The ext-v6 configuration block (campaign_spec_s53_w142_resettle_ext_v6_*), as the child reads it."""
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                              'ess_ageing_baseline': declared_ageing, 'ess_params_file': pin,
                              'ess_ageing_baseline_label': 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)',
                              'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL)},
            'cap': CAP, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE}


def parse_config(config):
    inst, model, flex = config.split('_')
    if inst not in INSTANCES or model not in MODELS or flex not in FLEX:
        raise ValueError(f'unknown configuration {config!r}; known {ALL_CONFIGS}')
    return inst, model, flex


def construct_configured(G, config, builder, ctx):
    inst, model, flex = parse_config(config)
    report, holder = {}, {}
    eval_id = f"p515s55_w183nl_{ctx['label']}_{ctx['run_tag']}_{config}_{builder}"
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(ctx['scratch'], config, builder), report, investment_map=INSTANCES[inst],
        eval_id=eval_id, num_max_iters_override=CAP, apply_rho=False, investment_year=INVESTMENT_YEAR)
    hook = H._config_hook_factory(ctx['spec'], holder, overrides={}, model_variant=MODELS[model],
                                  investment_year=INVESTMENT_YEAR, expected_floor_rows=ctx['floor_rows'],
                                  flex_price_multiplier=FLEX[flex], interface_deviation_premium=None,
                                  release_solution_bookkeeping=None)
    hook(planning=planning, sed=sed, candidate=candidate, report=report)
    checklist = report.get('rule_eleven_checklist') or {}
    readbacks = {
        'w20_model_variant': checklist.get('w20_model_variant'),
        'w33_flex_price_multiplier': checklist.get('w33_flex_price_multiplier'),
        'w21_ess_ageing_baseline': checklist.get('w21_ess_ageing_baseline'),
        'w84_convergence_depth_tail': checklist.get('w84_convergence_depth_tail'),
        's44_campaign_configuration_checks': checklist.get('s44_campaign_configuration_checks'),
    }
    state = {'parallel_execution': bool(planning.parallel_execution),
             'interface_deviation_premium': dict(planning.params.admm.interface_deviation_premium),
             'sed_ageing_enabled': getattr(sed, 'ageing_enabled', None),
             'sed_available_energy_soh_point': getattr(sed, 'available_energy_soh_point', None),
             'eval_id': eval_id, 'instance_report': report.get('instance')}
    return planning, sed, candidate, readbacks, state


def readbacks_ok(model, readbacks):
    ok = {}
    w20, w33, w21 = (readbacks.get('w20_model_variant'), readbacks.get('w33_flex_price_multiplier'),
                     readbacks.get('w21_ess_ageing_baseline'))
    if MODELS[model] is None:
        ok['w20_absent_for_baseline'] = w20 is None
        ok['w21_readback_all_match'] = bool(w21) and w21.get('readback_all_match') is True
    else:
        ok['w20_readback_all_match'] = bool(w20) and w20.get('readback_all_match') is True
        ok['w20_apply_checks_all_pass'] = bool(w20) and all(w20.get('apply_checks', {}).values())
        ok['w20_floor_rows_identical_to_baseline_probe'] = bool(w20) and w20.get(
            'floor_rows_identical_to_baseline_probe') is True
    ok['w21_checks_all_pass'] = bool(w21) and all((w21.get('checks') or {}).values())
    ok['w33_readback_all_match'] = bool(w33) and w33.get('readback_all_match') is True
    ok['w33_checks_all_pass'] = bool(w33) and all((w33.get('checks') or {}).values())
    ok['s44_configuration_checks_all_pass'] = all((readbacks.get('s44_campaign_configuration_checks') or {}).values())
    return ok


def build_config(G, srp, config, ctx):
    inst, model, flex = parse_config(config)
    record = {'config': config, 'instance': inst, 'model': model, 'flex': flex,
              'investment_map': {str(k): list(v) for k, v in INSTANCES[inst].items()},
              'model_variant': MODELS[model], 'flex_price_multiplier': FLEX[flex]}
    canonical = H.canonical_candidate(INSTANCES[inst], investment_year=INVESTMENT_YEAR)
    record['candidate_canonical'] = canonical
    record['candidate_key'] = H.candidate_key(canonical)
    record['eval_key_base'] = H.evaluation_key(
        record['candidate_key'], {}, case_file_aa=CASE_FILE_AA, model_variant=MODELS[model],
        ess_ageing_baseline=ctx['declared_ageing'], flex_price_multiplier=FLEX[flex],
        convergence_depth_tail=CONVERGENCE_DEPTH_TAIL)
    # ---- sequential builder: production's own `_run_operational_planning` initialisation, every solve stubbed
    t0 = time.time()
    planning, sed, candidate, readbacks, state = construct_configured(G, config, 'sequential', ctx)
    cap = Capture(ctx['out'], config, 'sequential')
    planning.transmission_network.optimize = cap.network_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = cap.network_stub(f'DSO{node_id}')
    sed.optimize = cap.esso_stub
    convergence, _results, _models, _sens, _prim, run_state = planning.run_operational_planning(
        type='distributed', candidate_solution=deepcopy(candidate), print_results=False, debug_flag=False,
        return_state=True)
    expected_calls = {'TSO': 1, 'ESSO': 1, **{f'DSO{n}': 1 for n in planning.distribution_networks}}
    seq = {'construct_state': state, 'readbacks': readbacks, 'readback_checks': readbacks_ok(model, readbacks),
           'stub_calls': dict(cap.calls), 'stub_calls_expected': expected_calls,
           'initialization_failed': run_state.get('initialization_failed'), 'convergence': convergence,
           'n_blocks': len(cap.blocks), 'blocks': dict(sorted(cap.blocks.items())), 'wall_s': time.time() - t0}
    seq['checks'] = {'stub_calls_exact': cap.calls == expected_calls,
                     'initialization_failed_is_true': run_state.get('initialization_failed') is True,
                     'n_blocks_is_51': len(cap.blocks) == 51,
                     'parallel_execution_off': state['parallel_execution'] is False,
                     **seq['readback_checks']}
    del planning, sed, candidate, _results, _models, run_state, cap
    gc.collect()
    # ---- parallel builder (the ProcessPool worker function), in-process, fresh planning object
    t0 = time.time()
    planning, sed, candidate, readbacks, state = construct_configured(G, config, 'parallel', ctx)
    cap = Capture(ctx['out'], config, 'parallel')
    premium = planning.params.admm.interface_deviation_premium
    returned = []
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        dn.optimize = cap.network_stub(f'DSO{node_id}')
        r_node, _res, _model = srp.create_distribution_network_model(
            node_id, dn, candidate['total_capacity'], premium['alpha'], premium['floor'])
        returned.append(r_node == node_id)
        del _res, _model
    expected_calls = {f'DSO{n}': 1 for n in planning.distribution_networks}
    par = {'construct_state': state, 'readbacks': readbacks, 'readback_checks': readbacks_ok(model, readbacks),
           'premium_passed': {'alpha': premium['alpha'], 'floor': premium['floor']},
           'stub_calls': dict(cap.calls), 'stub_calls_expected': expected_calls,
           'n_blocks': len(cap.blocks), 'blocks': dict(sorted(cap.blocks.items())), 'wall_s': time.time() - t0}
    par['checks'] = {'stub_calls_exact': cap.calls == expected_calls, 'returned_node_ids': all(returned),
                     'n_blocks_is_36': len(cap.blocks) == 36, **par['readback_checks']}
    del planning, sed, candidate, cap
    gc.collect()
    record['sequential'] = seq
    record['parallel'] = par
    return record


# ======================================================================================================================
#  comparisons
# ======================================================================================================================
def _load_items(rel):
    with gzip.open(os.path.join(REPO, rel), 'rt') as handle:
        return json.load(handle)


def compare_blocks(rec_a, rec_b, builder, item_detail=True):
    """Every block of `builder` in two configurations: NL identity (plain, labelled, row, col) and component digests;
    for every differing component, the per-index differences from the item files, with the constraint active flags."""
    ba, bb = rec_a[builder]['blocks'], rec_b[builder]['blocks']
    out = {'blocks_compared': len(set(ba) | set(bb)), 'blocks_missing': sorted(set(ba) ^ set(bb)), 'per_block': {}}
    nl_differ, digest_differ = [], []
    for key in sorted(set(ba) & set(bb)):
        a, b = ba[key], bb[key]
        na, nb = a['nl'], b['nl']
        nl_eq = {
            'plain': bool(na['plain'] and nb['plain'] and na['plain']['sha256'] == nb['plain']['sha256']),
            'symbolic_nl': bool(na['symbolic'] and nb['symbolic']
                                and na['symbolic']['nl']['sha256'] == nb['symbolic']['nl']['sha256']),
            'row': bool(na['symbolic'] and nb['symbolic']
                        and na['symbolic']['row']['sha256'] == nb['symbolic']['row']['sha256']),
            'col': bool(na['symbolic'] and nb['symbolic']
                        and na['symbolic']['col']['sha256'] == nb['symbolic']['col']['sha256']),
        }
        comps_diff = sorted(n for n in set(a['components']) | set(b['components'])
                            if a['components'].get(n) != b['components'].get(n))
        entry = {'nl_identical': nl_eq, 'block_digest_identical': a['sha256'] == b['sha256'],
                 'components_differing': comps_diff}
        if not nl_eq['plain']:
            nl_differ.append(key)
        if comps_diff:
            digest_differ.append(key)
            if item_detail:
                ia, ib = _load_items(a['items_gz']), _load_items(b['items_gz'])
                detail = {}
                for name in comps_diff:
                    xa, xb = ia['items'].get(name) or {}, ib['items'].get(name) or {}
                    idx = sorted(i for i in set(xa) | set(xb) if xa.get(i) != xb.get(i))
                    d = {'n_items_a': len(xa), 'n_items_b': len(xb), 'n_differing': len(idx),
                         'examples': [{'index': i, 'a': xa.get(i), 'b': xb.get(i)} for i in idx[:3]]}
                    if name.startswith('Constraint:'):
                        fa = ia['constraint_active'].get(name) or {}
                        fb = ib['constraint_active'].get(name) or {}
                        act = [i for i in idx if fa.get(i) is True or fb.get(i) is True]
                        d['n_differing_active_in_a_or_b'] = len(act)
                        d['every_differing_index_inactive_in_both'] = not act
                        d['active_differing_examples'] = act[:5]
                    elif name.startswith('Var:'):
                        d['first_field_note'] = 'state = (value, raw lb, raw ub, fixed)'
                    detail[name] = d
                entry['component_detail'] = detail
        out['per_block'][key] = entry
    out['blocks_nl_plain_differing'] = nl_differ
    out['blocks_digest_differing'] = digest_differ
    out['all_nl_plain_identical'] = not nl_differ and not out['blocks_missing']
    out['all_digests_identical'] = not digest_differ and not out['blocks_missing']
    out['components_differing_union'] = sorted({c for k in digest_differ
                                                for c in out['per_block'][k]['components_differing']})
    return out


def parse_nl(text):
    """Segment the (comment-stripped) NL text. Returns per-row signatures (C body, J segment, r line), the column
    count, and the non-row segments (header, b, k, x, d, O, G, V, S, F, L) as text, for a by-name comparison."""
    lines = _strip_nl_comments(text).split('\n')
    header = lines[:10]
    n_cons = int(header[1].split()[1])
    segs = []
    cur = None
    for line in lines[10:]:
        if line and line[0] in NL_SEGMENT_HEADS:
            cur = [line, []]
            segs.append(cur)
        elif cur is not None:
            cur[1].append(line)
    rows = {i: {'C': None, 'J': None, 'r': None} for i in range(n_cons)}
    other = {'header': '\n'.join(header)}
    for head, body in segs:
        tag = head[0]
        if tag == 'C':
            rows[int(head[1:].split()[0])]['C'] = '\n'.join(body)
        elif tag == 'J':
            rows[int(head[1:].split()[0])]['J'] = head + '\n' + '\n'.join(body)
        elif tag == 'r':
            for i, line in enumerate(body[:n_cons]):
                rows[i]['r'] = line
        else:
            name = tag if tag in 'bk' else head.split()[0]
            other[name] = head + '\n' + '\n'.join(body)
    return rows, other


def nl_row_analysis(rec_a, rec_b, builder, key):
    """Per-row comparison of one block's NL via the labelled .row/.col files."""
    na, nb = rec_a[builder]['blocks'][key]['nl'], rec_b[builder]['blocks'][key]['nl']

    def load(rec):
        with gzip.open(os.path.join(REPO, rec['symbolic']['nl']['gz']), 'rt') as h:
            text = h.read()
        with gzip.open(os.path.join(REPO, rec['symbolic']['row']['gz']), 'rt') as h:
            rows = h.read().splitlines()
        with gzip.open(os.path.join(REPO, rec['symbolic']['col']['gz']), 'rt') as h:
            cols = h.read().splitlines()
        return text, rows, cols

    ta, ra, ca = load(na)
    tb, rb, cb = load(nb)
    rows_a, other_a = parse_nl(ta)
    rows_b, other_b = parse_nl(tb)
    by_name_a = {ra[i]: rows_a[i] for i in rows_a}
    by_name_b = {rb[i]: rows_b[i] for i in rows_b}
    names = sorted(set(by_name_a) | set(by_name_b))
    differing = [n for n in names if by_name_a.get(n) != by_name_b.get(n)]
    seg_diff = {n: sorted(s for s in ('C', 'J', 'r') if (by_name_a.get(n) or {}).get(s) != (by_name_b.get(n) or {}).get(s))
                for n in differing}
    other_diff = sorted(s for s in set(other_a) | set(other_b) if other_a.get(s) != other_b.get(s))
    header_lines = [{'line': i + 1, 'a': x, 'b': y} for i, (x, y) in enumerate(
        zip(other_a['header'].split('\n'), other_b['header'].split('\n'))) if x != y]
    k_detail = None
    if 'k' in other_diff:
        ka, kb = other_a['k'].split('\n')[1:], other_b['k'].split('\n')[1:]
        # the k segment holds cumulative counts of columns 0..n-2; the last column's count is the header's Jacobian
        # nonzero total (line 8, first field) minus the last cumulative count
        cum_a = [int(v) for v in ka if v.strip()] + [int(other_a['header'].split('\n')[7].split()[0])]
        cum_b = [int(v) for v in kb if v.strip()] + [int(other_b['header'].split('\n')[7].split()[0])]
        per_a = [cum_a[0]] + [cum_a[j] - cum_a[j - 1] for j in range(1, len(cum_a))]
        per_b = [cum_b[0]] + [cum_b[j] - cum_b[j - 1] for j in range(1, len(cum_b))]
        cols_changed = [j for j in range(min(len(per_a), len(per_b))) if per_a[j] != per_b[j]]
        k_detail = {'note': ('per-column Jacobian nonzero counts from the k segment (cumulative, columns 0..n-2) and '
                             'the header total (last column)'),
                    'n_columns_with_changed_count': len(cols_changed),
                    'columns_changed': [{'col': ca[j] if j < len(ca) else j, 'count_a': per_a[j],
                                         'count_b': per_b[j]} for j in cols_changed[:20]]}
    return {'block': key, 'col_identical': ca == cb, 'row_names_identical': ra == rb,
            'n_rows_a': len(rows_a), 'n_rows_b': len(rows_b), 'n_cols_a': len(ca), 'n_cols_b': len(cb),
            'rows_with_differing_segments': differing, 'n_rows_with_differing_segments': len(differing),
            'segments_differing_by_row': seg_diff,
            'every_differing_row_is_energy_storage_capacity_degradation': all(
                DEGRADATION_ROW_RE.match(n) for n in differing),
            'non_row_segments_differing': other_diff, 'header_lines_differing': header_lines,
            'k_segment_detail': k_detail}


def pair(records, a, b):
    return a in records and b in records


def comparisons(records):
    out = {}
    builders = ('sequential', 'parallel')
    # (i) x0, A vs N
    for flex in FLEX:
        ca, cb = f'x0_A_{flex}', f'x0_N_{flex}'
        if not pair(records, ca, cb):
            continue
        out[f'i_{ca}_vs_{cb}'] = {b: compare_blocks(records[ca], records[cb], b) for b in builders}
    # (ii) unit, A vs N (+ the per-row analysis of every ESSO NL that differs)
    for flex in FLEX:
        ca, cb = f'unit_A_{flex}', f'unit_N_{flex}'
        if not pair(records, ca, cb):
            continue
        res = {b: compare_blocks(records[ca], records[cb], b) for b in builders}
        res['esso_nl_row_analysis'] = {k: nl_row_analysis(records[ca], records[cb], 'sequential', k)
                                       for k in res['sequential']['blocks_nl_plain_differing']
                                       if k.startswith('ESSO')}
        out[f'ii_{ca}_vs_{cb}'] = res
    # (iii) 1x vs 2x within (instance, model)
    for inst in INSTANCES:
        for model in MODELS:
            ca, cb = f'{inst}_{model}_1x', f'{inst}_{model}_2x'
            if not pair(records, ca, cb):
                continue
            out[f'iii_{ca}_vs_{cb}'] = {b: compare_blocks(records[ca], records[cb], b) for b in builders}
    return out


def summarize(comp, records):
    """The four answers, mechanically (no verdicts beyond these)."""
    s = {}
    for name, res in comp.items():
        row = {}
        for b in ('sequential', 'parallel'):
            r = res[b]
            net = [k for k in r['per_block'] if not k.startswith('ESSO')]
            esso = [k for k in r['per_block'] if k.startswith('ESSO')]
            row[b] = {
                'network_blocks': len(net),
                'network_nl_plain_identical_all': all(r['per_block'][k]['nl_identical']['plain'] for k in net),
                'network_nl_all_four_files_identical_all': all(all(r['per_block'][k]['nl_identical'].values())
                                                               for k in net),
                'network_digest_identical_all': all(not r['per_block'][k]['components_differing'] for k in net),
                'network_components_differing_union': sorted({c for k in net
                                                              for c in r['per_block'][k]['components_differing']}),
                'network_blocks_nl_differing': sorted(k for k in net if not r['per_block'][k]['nl_identical']['plain']),
            }
            if esso:
                row[b]['esso_nl_plain_identical_by_node'] = {k: r['per_block'][k]['nl_identical']['plain'] for k in esso}
                row[b]['esso_nl_all_four_files_identical_by_node'] = {
                    k: all(r['per_block'][k]['nl_identical'].values()) for k in esso}
                row[b]['esso_components_differing_by_node'] = {k: r['per_block'][k]['components_differing']
                                                               for k in esso}
                inactive = {}
                for k in esso:
                    det = r['per_block'][k].get('component_detail') or {}
                    inactive[k] = {n: d.get('every_differing_index_inactive_in_both')
                                   for n, d in det.items() if n.startswith('Constraint:')}
                row[b]['esso_differing_constraints_every_index_inactive'] = inactive
        if 'esso_nl_row_analysis' in res:
            row['esso_nl_row_analysis'] = {k: {kk: v[kk] for kk in (
                'col_identical', 'row_names_identical', 'n_rows_with_differing_segments',
                'every_differing_row_is_energy_storage_capacity_degradation', 'non_row_segments_differing')}
                for k, v in res['esso_nl_row_analysis'].items()}
        s[name] = row
    keys = {c: r['eval_key_base'] for c, r in records.items()}
    s['eval_keys'] = keys
    s['eval_keys_distinct'] = len(set(keys.values())) == len(keys)
    return s


# ======================================================================================================================
#  provenance and capture-path assertion (rule eleven: before anything is built)
# ======================================================================================================================
def production_modules():
    out = _git(['ls-files', '*.py'])
    return sorted(f for f in out.splitlines() if '/' not in f and not re.match(r'p\d', f))


def input_files():
    out = _git(['ls-files', 'data/SRP1'])
    return sorted(f for f in out.splitlines()
                  if not f.startswith(('data/SRP1/Results/', 'data/SRP1/Diagrams/')))


def repo_modules_imported():
    out = {}
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, '__file__', None)
        if not path:
            continue
        path = os.path.abspath(path)
        if path.startswith(REPO + os.sep) and path.endswith('.py') and '/.claude/' not in path:
            out[os.path.relpath(path, REPO)] = _sha_file(path)
    return dict(sorted(out.items()))


def git_status_of(paths):
    if not paths:
        return {}
    text = _git(['status', '--porcelain', '--'] + list(paths))
    return {line[3:]: line[:2] for line in text.splitlines()}


def assert_capture_paths(G, srp, PG):
    import pyomo.repn.plugins.nl_writer as NLW
    precedent_src = open(os.path.join(REPO, PRECEDENT_REL)).read()
    tree = ast.parse(precedent_src)
    precedent_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'block_state')
    precedent_text = ast.get_source_segment(precedent_src, precedent_fn)
    hook_params = inspect.signature(H._config_hook_factory).parameters
    checks = {
        'block_state_verbatim_from_precedent': inspect.getsource(block_state).rstrip() == precedent_text.rstrip(),
        'construct_arm_planning_exists': callable(getattr(G, '_construct_arm_planning', None)),
        'config_hook_takes_model_variant_and_flex': {'model_variant', 'flex_price_multiplier'} <= set(hook_params),
        'parallel_worker_signature': list(inspect.signature(srp.create_distribution_network_model).parameters)
        == ['node_id', 'distribution_network', 'candidate_solution', 'premium_alpha', 'premium_floor'],
        'init_failed_branch_returns_before_update_to_admm': (
            "'initialization_failed': True" in inspect.getsource(srp._run_operational_planning)),
        'nl_writer_call_forces_presolve_and_scaling_off': ('config.linear_presolve = False'
                                                           in inspect.getsource(NLW.NLWriter.__call__)
                                                           and 'config.scale_model = False'
                                                           in inspect.getsource(NLW.NLWriter.__call__)),
        'floor_rows_builder_exists': callable(getattr(PG, '_build_floor_rows', None)),
        'evaluation_key_takes_tail': 'convergence_depth_tail' in inspect.signature(H.evaluation_key).parameters,
    }
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise AssertionError(f'RULE ELEVEN: capture paths missing: {failed}')
    return checks


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='output dir (repo-relative or absolute); write-once')
    parser.add_argument('--label', required=True)
    parser.add_argument('--scratch', required=True, help='scratch dir for the planning results redirect (outside the '
                                                          'repo); must not exist')
    parser.add_argument('--configs', default=','.join(ALL_CONFIGS))
    parser.add_argument('--determinism-configs', default=DEFAULT_DETERMINISM)
    parser.add_argument('--no-determinism', action='store_true')
    parser.add_argument('--expect-production-commit', default=None)
    parser.add_argument('--allow-uncommitted-harness', action='store_true',
                        help='smoke tests only: do not refuse when this script is not committed and clean')
    args = parser.parse_args()
    started = time.time()
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    configs = [c for c in args.configs.split(',') if c]
    for c in configs:
        parse_config(c)
    det_configs = [] if args.no_determinism else [c for c in args.determinism_configs.split(',') if c]
    for c in det_configs:
        parse_config(c)
        if c not in configs:
            raise SystemExit(f'determinism configuration {c} is not in --configs')
    failures = []
    if os.path.exists(out):
        present = sorted(os.listdir(out))
        if [p for p in present if p != 'launch.log']:
            failures.append(f'output directory exists and is not empty (write-once): {out}: {present[:5]}')
    if os.path.exists(args.scratch):
        failures.append(f'scratch dir exists: {args.scratch}')
    if os.path.abspath(args.scratch).startswith(REPO + os.sep):
        failures.append(f'scratch dir must be outside the repo: {args.scratch}')
    head = _git(['rev-parse', 'HEAD']).strip()
    prod = production_modules()
    prod_status = git_status_of(prod)
    if prod_status:
        failures.append(f'production modules not clean: {prod_status}')
    expected_commit = None
    if args.expect_production_commit:
        expected_commit = _git(['rev-parse', args.expect_production_commit]).strip()
        diff = subprocess.run(['git', 'diff', '--quiet', expected_commit, '--'] + prod, cwd=REPO).returncode
        if diff != 0:
            failures.append(f'production modules differ from {expected_commit}')
    self_status = git_status_of([SCRIPT_REL])
    self_tracked = bool(_git(['ls-files', SCRIPT_REL]).strip())
    if not args.allow_uncommitted_harness and (self_status or not self_tracked):
        failures.append(f'{SCRIPT_REL} is not committed and clean ({self_status or "untracked"})')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out, exist_ok=True)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    _log(f'{STAGE}; label {args.label}; HEAD {head}; configs {configs}; determinism {det_configs or "none"}')

    import pyomo
    import p515_g_g1_g4_admm_gates as G
    import p515_s40_polish_gap as PG
    import shared_resources_planning as srp
    capture_checks = assert_capture_paths(G, srp, PG)
    _log(f'rule eleven capture-path checks: {capture_checks}')
    ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
    ess_sha = _sha_file(ess_path)
    declared = H.load_ess_ageing_parameters(ess_path)
    pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': ess_sha}
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s55_w183nl_{args.label}_{run_tag}_precheck')
    os.makedirs(args.scratch)
    ctx = {'out': out, 'label': args.label, 'run_tag': run_tag, 'scratch': args.scratch,
           'spec': spec_like(declared, pin), 'declared_ageing': declared, 'floor_rows': floor_rows}
    modules_start = repo_modules_imported()

    records = {}
    for config in configs:
        t0 = time.time()
        records[config] = build_config(G, srp, config, ctx)
        seq, par = records[config]['sequential'], records[config]['parallel']
        _log(f"{config}: eval key {records[config]['eval_key_base'][:16]}; sequential {seq['n_blocks']} blocks "
             f"checks {all(seq['checks'].values())} ({seq['wall_s']:.0f}s); parallel {par['n_blocks']} blocks "
             f"checks {all(par['checks'].values())} ({par['wall_s']:.0f}s); total {time.time() - t0:.0f}s")
        for b in ('sequential', 'parallel'):
            bad = [k for k, v in records[config][b]['checks'].items() if not v]
            if bad:
                _log(f'  [CHECK FAILED] {config} {b}: {bad}')
            esso = {k: v['nl']['plain']['sha256'][:12] for k, v in records[config][b]['blocks'].items()
                    if k.startswith('ESSO') and v['nl']['plain']}
            if esso:
                _log(f'  ESSO plain NL: {esso}')

    comp = comparisons(records)
    summary = summarize(comp, records)
    for name, row in summary.items():
        if name.startswith(('i_', 'ii_', 'iii_')):
            _log(f'{name}: {json.dumps(row, default=str)[:1500]}')

    determinism = None
    if det_configs:
        child_out = os.path.join(out, 'determinism_child')
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--out', child_out, '--label', f'{args.label}_det',
               '--scratch', os.path.join(args.scratch + '_det'), '--configs', ','.join(det_configs),
               '--no-determinism']
        if args.expect_production_commit:
            cmd += ['--expect-production-commit', args.expect_production_commit]
        if args.allow_uncommitted_harness:
            cmd += ['--allow-uncommitted-harness']
        _log(f'(iv) determinism: separate process {cmd}')
        sys.stdout.flush()
        sys.stderr.flush()
        rc = subprocess.run(cmd, cwd=REPO).returncode
        child_json = os.path.join(child_out, 'w183_nl_digest.json')
        determinism = {'command': cmd, 'returncode': rc, 'configs': det_configs}
        if os.path.isfile(child_json):  # compared whenever the child wrote its record; its exit code is a check
            with open(child_json) as handle:
                child = json.load(handle)
            determinism['child_json_sha256'] = _sha_file(child_json)
            determinism['child_guard'] = child.get('solve_profile_guard')
            per = {}
            for c in det_configs:
                per[c] = {}
                for b in ('sequential', 'parallel'):
                    ra, rb = records[c][b]['blocks'], child['records'][c][b]['blocks']
                    diffs = []
                    for key in sorted(set(ra) | set(rb)):
                        a, bb = ra.get(key), rb.get(key)
                        if a is None or bb is None:
                            diffs.append({'block': key, 'missing': True})
                            continue
                        files = {'plain': (a['nl']['plain'] or {}).get('sha256') == (bb['nl']['plain'] or {}).get('sha256')}
                        for f in ('nl', 'row', 'col'):
                            files[f'symbolic_{f}'] = ((a['nl']['symbolic'] or {}).get(f, {}).get('sha256')
                                                      == (bb['nl']['symbolic'] or {}).get(f, {}).get('sha256'))
                        files['block_digest'] = a['sha256'] == bb['sha256']
                        files['component_digests'] = a['components'] == bb['components']
                        if not all(files.values()):
                            diffs.append({'block': key, 'files': files})
                    per[c][b] = {'n_blocks': len(set(ra) | set(rb)), 'differences': diffs,
                                 'byte_identical': not diffs}
            determinism['per_config'] = per
            determinism['byte_identical_all'] = all(v['byte_identical'] for c in per.values() for v in c.values())
            determinism['child_all_ok'] = child.get('all_ok')
            determinism['child_checks'] = child.get('checks')
        else:
            determinism['byte_identical_all'] = False
            determinism['error'] = f'child exit {rc} or no JSON at {child_json}'
        _log(f"(iv) determinism: byte-identical {determinism.get('byte_identical_all')} "
             f"(child exit {rc})")

    modules_end = repo_modules_imported()
    checks = {
        'capture_paths_asserted': all(capture_checks.values()),
        'every_build_check_passes': all(all(records[c][b]['checks'].values()) for c in records
                                        for b in ('sequential', 'parallel')),
        'no_nl_write_errors': all(v['nl']['error'] is None for c in records for b in ('sequential', 'parallel')
                                  for v in records[c][b]['blocks'].values()),
        'labelled_nl_equals_plain_after_comment_strip': all(
            v['nl']['stripped_equal'] is True for c in records for b in ('sequential', 'parallel')
            for v in records[c][b]['blocks'].values()),
        'imported_repo_modules_unchanged_during_run': modules_start == {k: modules_end.get(k) for k in modules_start},
    }
    unit_n = [c for c in records if c.startswith('unit_N_')]
    if unit_n:
        checks['eval_key_unit_N_1x_equals_ext_v6_base_key'] = (
            'unit_N_1x' not in records or records['unit_N_1x']['eval_key_base'] == EXT_V6_UNIT_N_BASE_KEY)
    keys = [records[c]['eval_key_base'] for c in records]
    checks['eval_keys_pairwise_distinct'] = len(set(keys)) == len(keys)
    if determinism is not None:
        checks['determinism_byte_identical'] = determinism.get('byte_identical_all') is True
        checks['determinism_child_exit_0'] = determinism.get('returncode') == 0
    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    prod_hashes = {f: _sha_file(os.path.join(REPO, f)) for f in prod}
    input_hashes = {f: _sha_file(os.path.join(REPO, f)) for f in input_files()}
    payload = {
        'stage': STAGE, 'authority': AUTHORITY, 'label': args.label, 'timestamp_utc': _utc(), 'run_tag': run_tag,
        'argv': sys.argv, 'git_head_at_run': head, 'script': SCRIPT_REL, 'script_sha256': _sha_file(__file__),
        'script_git_status': self_status or ('tracked, clean' if self_tracked else 'untracked'),
        'allow_uncommitted_harness': args.allow_uncommitted_harness,
        'production': {'definition': 'tracked root-level *.py not matching ^p\\d', 'modules_sha256': prod_hashes,
                       'expected_commit': expected_commit,
                       'equal_to_expected_commit': expected_commit is not None,
                       'last_commit_touching': _git(['log', '-1', '--format=%H', '--'] + prod).strip()},
        'repo_modules_imported_sha256_start': modules_start, 'repo_modules_imported_sha256_end': modules_end,
        'harness_files_status': git_status_of(list(modules_start)),
        'inputs': {'definition': 'every tracked file under data/SRP1 outside Results/ and Diagrams/',
                   'sha256': input_hashes, 'git_status': git_status_of(list(input_hashes)),
                   'ess_params_pin': pin},
        'pyomo_version': pyomo.version.version, 'python': sys.version, 'interpreter': sys.executable,
        'env': {k: os.environ.get(k) for k in ('NLP_SOLVER_PATH', 'LP_SOLVER_PATH', 'OMP_NUM_THREADS')},
        'configuration': {'spec_like': ctx['spec'], 'cap': CAP, 'investment_year': INVESTMENT_YEAR,
                          'instances': {k: {str(n): list(v) for n, v in m.items()} for k, m in INSTANCES.items()},
                          'models': MODELS, 'flex': FLEX, 'configs': configs,
                          'floor_rows_by_node': floor_rows},
        'capture_path_checks': capture_checks,
        'digest_definition': ('per block, at the would-be solve: sha256 over per-component sha256 of {Param values; '
                              'Var (value, raw lb, raw ub, fixed); Constraint (active, lower, upper, str(body)); '
                              'Objective (active, str(expr)); Expression str(expr); block active flags} -- '
                              '`block_state`, verbatim from p515_s47_case_file_baseline_check.py. NL: '
                              'Block.write(format=nl) with symbolic_solver_labels False (production form; '
                              'NLWriter.__call__ forces linear_presolve / scale_model off) and True (+ .row/.col); '
                              'sha256 of the raw files'),
        'records': records, 'comparisons': comp, 'summary': summary, 'determinism': determinism,
        'checks': checks, 'all_ok': all_ok,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started}
    path = os.path.join(out, 'w183_nl_digest.json')
    with open(path, 'w') as handle:
        GRIO.dump(_str_keys(payload), handle, indent=1, sort_keys=True)
    manifest = {}
    for root, _dirs, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.relpath(p, out) in ('launch.log', 'manifest_sha256.json'):
                continue
            manifest[os.path.relpath(p, REPO)] = _sha_file(p)
    with open(os.path.join(out, 'manifest_sha256.json'), 'w') as handle:
        GRIO.dump(dict(sorted(manifest.items())), handle, indent=1, sort_keys=True)
    shutil.rmtree(args.scratch, ignore_errors=True)
    GUARD.uninstall()
    for k, v in checks.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    _log(f'wrote {path} sha256={_sha_file(path)}; manifest {len(manifest)} files')
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
