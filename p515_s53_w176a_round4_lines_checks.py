"""
P5.15 Addendum 72 order, Planner task W176a -- static checks behind the re-check of the round-4 lines at Overleaf
8b76423 (P5_15_W176A_ROUND4_LINES.md). ZERO SOLVES, NO MODEL, NO PICKLE, no production module imported.

What it does (static reading only):
  A. Manuscript identity: the Overleaf clone's HEAD (declared 8b76423), main.tex sha256 (declared, full hash) and line
     count (2,156), response_to_reviewers_draft.tex sha256 (declared prefix b7a269d5), no file of the clone modified.
  B. Round-4 presence: every find -> replace of STEP6_ROUND4_CORRECTIONS.md (items 1-14) and the l. 858 reference,
     transcribed below; each transcription is first asserted to occur in the corrections file (whitespace-collapsed),
     except the chemistry texts, which are Addendum 72's quotations. Each new text is searched in main.tex (or the
     letter) with ALL whitespace removed (presence + line numbers); each old text is checked absent. Item 9's new
     text is the corrections file's ```latex block (parsed). Item 14's k <- k + 1 placement is checked structurally.
  C. Literal "Section~4.x" references (and "Section 4.x", "Sec.~4", "Subsection~4.x", "Sections~4") in sections 2-3
     of main.tex, by regex, with the section ranges taken from the \\section lines; every occurrence elsewhere in
     main.tex is listed too. "[CONFIRM" and "[AUTHOR]" counts in main.tex, the letter, the cover letter, the
     highlights and the uncompiled section2_expert_draft.tex.
  D. Record checks: D1 the three search campaigns (termination, certificate recorded or null, direction
     admissibility per poll, evaluated or refused); D2 the search's acceptance threshold at every evaluated poll point
     against the determinacy thresholds of the frozen claims (coarser or not); D3 the tight tail and the code identity
     of every campaign that produced a certified cell of the frozen tables, and of the three searches; D4 the C4 arm of
     ext spec v3 (entered triple, k recomputed); D5 the benchmark (instance, the six NRF run records: consistency-pass
     trigger per DN block, cost source, tie-breakers); D6 the chemistry strings of the ESS input files; D7 Pyomo's
     mapping of IPOPT exit codes (acceptable -> optimal).
  E. Code anchors: exact-text search of each code location the report cites (every anchor must be found).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit; pyomo is imported by the guard only). Refuses to overwrite an existing output.

Output (new directory): data/SRP1/Results/P515S53/w176a_round4_lines/w176a_checks.json and manifest_sha256.json.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S53/w176a_round4_lines && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w176a_round4_lines_checks.py \\
      > data/SRP1/Results/P515S53/w176a_round4_lines/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W176a round-4 line checks (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W176a: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W176a: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w176a_round4_lines')
OUT_JSON = os.path.join(OUT_DIR, 'w176a_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
COVER = os.path.join(CLONE, 'cover_letter.tex')
HIGHLIGHTS = os.path.join(CLONE, 'highlights.tex')
DRAFT2 = os.path.join(CLONE, 'section2_expert_draft.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
DECLARED_CLONE_HEAD = '8b76423'
DECLARED_MAIN_SHA256 = '7aa9e105deccdf61cced665f03d95b02ea736fe5c34f3402f6281db89d19624b'
DECLARED_MAIN_LINES = 2156
DECLARED_LETTER_SHA256_PREFIX = 'b7a269d5'
ROUND4 = 'STEP6_ROUND4_CORRECTIONS.md'

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
FROZEN_TABLES = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
EXT_SPEC_V3 = os.path.join(P53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
BENCH_SPEC_V5 = os.path.join(P53, 'w116_benchmark_nrf', 'frozen_s53_benchmark_spec_v5_bca69f97.json')
NRF_RUNS = sorted(glob.glob(os.path.join(P53, 'w116_benchmark_nrf', 'nrf_arm_*_r2', 'nrf_arm_*_r2.json')))
ESS_FILES = (os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
             os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_params.json'))
SEARCH = {
    's47': {'results': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b', 'campaign_results.json'),
            'spec': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                                 'campaign_spec_s47_phase_b_8cfa264e.json'), 'budget': 20},
    's51': {'results': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b', 'campaign_results.json'),
            'spec': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                                 'campaign_spec_s51_f2_phase_b_5ce295e1.json'), 'budget': 20},
    's53': {'results': os.path.join(P53, 'campaign_s53_f2_certificate_r1', 'campaign_results.json'),
            'spec': os.path.join(P53, 'campaign_s53_f2_certificate_r1', 'campaign_spec_s53_f2_certificate_r1_803571c0.json'),
            'budget': 60},
}
PRODUCTION_FILES = ('admm_anderson_acceleration.py', 'admm_parameters.py', 'admm_persistent_workers.py', 'branch.py',
                    'definitions.py', 'energy_storage.py', 'generator.py', 'helper_functions.py', 'load.py',
                    'model_construction_helpers.py', 'network.py', 'network_data.py', 'network_parameters.py',
                    'node.py', 'planning_parameters.py', 'shared_energy_storage.py', 'shared_energy_storage_data.py',
                    'shared_energy_storage_parameters.py', 'shared_resources_planning.py', 'solver_parameters.py')
CASE_DATA = (os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_params.json'),
             os.path.join('data', 'SRP1', 'SharedESS'), os.path.join('data', 'SRP1', 'case9'),
             os.path.join('data', 'SRP1', 'case33_1'), os.path.join('data', 'SRP1', 'case33_2'),
             os.path.join('data', 'SRP1', 'case33_3'), os.path.join('data', 'SRP1', 'MarketData'))
CONTROL_HEAD = '353e094b'
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions',
                             'uncoordinated_benchmark')
INPUTS = set()

# ---------------------------------------------------------------------------------------------------------------------
#  B. round-4 items, transcribed verbatim from STEP6_ROUND4_CORRECTIONS.md (asserted below) -- (id, file, new, old)
#  `new` None = only the old text's absence is checked; `old` None = no old text named.
# ---------------------------------------------------------------------------------------------------------------------
ITEMS = [
    ('1a', 'main', 'in which every rounded poll direction was inadmissible at every evaluated poll, so that',
     'in which every rounded poll direction was inadmissible at every poll, so that'),
    ('1b', 'main', 'of the three runs reported here, two stopped when a unit poll failed and one was stopped for review '
                   'when its completion set exceeded the cap; each is described by its recorded certificate.',
     'the runs reported here stopped at their evaluation budgets and are described by their recorded certificates.'),
    ('2', 'main', '; smaller differences are ties, reported as ``within resolution\'\'. The master search of '
                  'Algorithm~\\ref{alg:shared_ess_planning_mads} accepted its poll points by the coarser resolution rule '
                  'stated there; every cell it proposed was re-evaluated under this rule before being reported.',
     'and the master search of Algorithm~\\ref{alg:shared_ess_planning_mads} accepts a poll point as an improvement '
     'only by a determinate margin; smaller differences are ties, reported as ``within resolution\'\'.'),
    ('3', 'main', 'at the end of the block', 'at the end of the representative year'),
    ('4a (l. 748)', 'main', 'The values of $\\eta$, $SoC^{\\text{Min}}$, $SoC^{\\text{Max}}$, $SoC^{0}$, '
                            '$\\varepsilon^{\\text{Cl}}$, $c^{\\text{Cl}}$ and $\\varepsilon^{\\text{C}}$ are given in '
                            'Section~\\ref{sec:case_ess_params}.',
     'The values of $\\eta$, $SoC^{\\text{Min}}$, $SoC^{\\text{Max}}$, $SoC^{0}$, $\\varepsilon^{\\text{Cl}}$, '
     '$c^{\\text{Cl}}$ and $\\varepsilon^{\\text{C}}$ are given in Section~\\ref{sec:case_settings}.'),
    ('4b (l. 858)', 'main', 'The values of $c^{\\sigma}$ and $\\varepsilon^{\\text{E}}$ are given in '
                            'Section~\\ref{sec:case_ess_params}.',
     'The values of $c^{\\sigma}$ and $\\varepsilon^{\\text{E}}$ are given in Section~\\ref{sec:case_settings}.'),
    ('5', 'main', '\\eqref{eq:soh_chain}, the available-energy product', '\\eqref{eq:soh_chain} the available-energy product'),
    ('6a (sec. 3.5)', 'main', 'utility-scale lithium-ion battery (the NREL ATB utility-scale battery '
                              'category~\\cite{nrel_ess_costs})', 'utility-scale lithium iron phosphate battery'),
    ('6b (letter R1.2(iv))', 'letter', 'the reference technology is utility-scale lithium-ion battery storage; the '
                                       'cycling calibrations are datasheet readings (Section~3.5)',
     'lithium iron phosphate (LFP)'),
    ('6c (abstract)', 'main', 'utility-scale lithium-ion battery storage', None),
    ('7a (row)', 'main', 'Datasheet 8\\,000 cycles$^{a}$ & (8\\,000, 1.00, 0.70)', None),
    ('7b (caption)', 'main', '$^{a}$~entered as 10\\,000 cycles at 0.80 depth of discharge, the same product '
                             '$N^{\\text{DS}} \\delta^{\\text{DS}}$, which is all \\eqref{eq:cycle_life_calibration} uses.',
     None),
    ('8', 'main', 'Every certified evaluation reported in this paper ran the coordination procedure of '
                  '\\ref{app:admm_updated} under one frozen configuration; the three planning-search campaigns that '
                  'proposed the incumbents ran on earlier states of the code, without the tight tail, and their '
                  'incumbents were re-evaluated under this configuration before being reported.',
     'Every recourse evaluation ran the coordination procedure of \\ref{app:admm_updated} under one frozen '
     'configuration.'),
    ('9', 'main', None, 'Each arrangement is reported as the best of three solver starts.'),  # new = latex block
    ('10', 'main', 'that is Gauss--Seidel on the interface channels and, on the storage channel, a global-variable '
                   'consensus in which all three agents solve against the same $z$ before it is updated; the penalty '
                   'parameters follow residual balancing, and the iteration uses Anderson acceleration and a tightened '
                   'interior-point tail.',
     'before it is updated with residual balancing of the penalty parameters, Anderson acceleration and a tightened '
     'interior-point tail.'),
    ('11', 'main', 'which is equivalent to dividing the agent\'s local objective by $\\kappa^{\\text{E}}$ and puts it on '
                   'the same footing as a median block\'s scaled objective.',
     'the consensus terms themselves are unscaled in every agent.'),
    ('12', 'main', 'every local solve of the cycle ended at a status the solver reports as solved (optimal or acceptable)',
     'every local solve of the cycle ended at an optimal status'),
    ('13a', 'main', 'The memory is cleared on any change of a penalty parameter and on any failed local solve, and the '
                    'acceleration is switched off on any cycle in which every channel passes and, in the certifying '
                    'regime, from the cycle after $k_0$ on (or after the earlier run\'s stopping cycle in a continued '
                    'evaluation).',
     'The memory is cleared on any change of a penalty parameter and on any failed local solve, from the cycle after '
     '$k_0$ on'),
    ('13b', 'main', '(the tail is a declared option, enabled in every campaign behind the reported tables)',
     '(the tail is a declared option, enabled in every campaign)'),
    ('14a', 'main', '(the interface settlement\'s weight is set to one here and, in the multi-scenario instance, the '
                    'commitment charge is activated)',
     '(in the multi-scenario instance the commitment charge and the settlement are activated here)'),
    ('14b', 'main', '$k \\gets k + 1$\\;', None),
]
# texts not quoted verbatim in the corrections file (transcription assertion skipped, source stated): 4a/4b the full
# sentences around item 4's reference swap (Addendum 72 quotes the l. 858 one); 6c the abstract wording quoted in
# Addendum 72 (the author's second chemistry option); 9 the old sentence (W175's quotation of the 260bd83 paragraph)
EXEMPT_TRANSCRIPTION = {'4a (l. 748)', '4b (l. 858)', '6c (abstract)', '9'}
# item 9 also: the sentence that opened the replaced range must survive as the start of the new block
ITEM9_OPEN = 'In both, the TSO then dispatches the TN with the interface exchanges fixed at the DNs\' schedules'

CODE_ANCHORS = [
    ('search_acceptance', 'p515_s47_phase_b_record.py', 'max(bar_x + bar_inc, sigma_q)'),
    ('s47_delta0', 'p515_s47_phase_b_record.py', 'DELTA_0 = 4'),
    ('budget_refusal', 'p515_s47_phase_b_record.py', 'evaluation_budget_exhausted'),
    ('soh_available_end_row', 'shared_energy_storage_data.py',
     'model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y]'),
    ('soh_available_mid_row', 'shared_energy_storage_data.py',
     "soh_mid = prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y] / 2.00)"),
    ('k_formula', 'shared_energy_storage_parameters.py', 'return self.cycles_n * self.reference_dod_d / (-log(self.eol_retention_r))'),
    ('cl_nom_default', 'shared_energy_storage_parameters.py', 'self.cl_nom = 10000'),
    ('settlement_weight_dso', 'shared_resources_planning.py', 'dso_model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('settlement_weight_tso', 'shared_resources_planning.py', 'model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('row18_with_settlement', 'shared_resources_planning.py', '_activate_row18_with_settlement(dso_model[year][day])'),
    ('prepare_then_convert_dso', 'shared_resources_planning.py', '_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)'),
    ('convert_dso', 'shared_resources_planning.py', 'update_distribution_models_to_admm(planning_problem, dso_models, admm_parameters, objective_scale)'),
    ('esso_al_scale', 'shared_resources_planning.py', 'admm_esso_al_scale'),
    ('local_solves_succeeded', 'shared_resources_planning.py', 'def _admm_local_solves_succeeded(planning_problem, results):'),
    ('solver_result_succeeded', 'helper_functions.py', 'def solver_result_succeeded(result):'),
    ('accepted_terminations', 'helper_functions.py', 'po.TerminationCondition.locallyOptimal,'),
    ('aa_off_all_pass', 'admm_anderson_acceleration.py', "action='off (all channels within Boyd tolerance)'"),
    ('hold_after_first_pass', 'p515_s53_w118_resettle_hooks.py', 'return self.first_pass is not None and c > self.first_pass'),
    ('aa_forced_off_in_hold', 'p515_s53_w118_resettle_hooks.py', "forced['all_boyd_pass'] = True"),
    ('tail_default_off', 'admm_parameters.py', 'self.convergence_depth_tail = {'),
    ('bench_single_scenario', 'uncoordinated_benchmark.py', 'def require_single_scenario(planning_problem):'),
    ('bench_global_trigger', 'uncoordinated_benchmark.py', 'trigger = trigger or violations[\'trigger_sequential_pass\']'),
    ('bench_reeval_every_block', 'uncoordinated_benchmark.py', 'def reevaluate_dso_at_actual_voltage('),
    ('bench_solve_all_dso', 'uncoordinated_benchmark.py', 'def solve_dso_models(planning_problem, dso_models, *, phase, record_callback=None):'),
    ('harness_pass_trigger', 'p515_s53_w116_benchmark_nrf.py', "if reeval['trigger_sequential_pass']:"),
    ('harness_pass_all_dso', 'p515_s53_w116_benchmark_nrf.py',
     "UB.solve_dso_models(planning, arm_out['models']['dso'], phase='sequential_pass:dso', record_callback=sink)"),
    ('harness_cost_after_pass', 'p515_s53_w116_benchmark_nrf.py', "arm_cost_source = 'phase_C_sequential_pass'"),
]


# ---------------------------------------------------------------------------------------------------------------------
def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    INPUTS.add(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, check=True).stdout


def _walk(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _strip(t):
    return ''.join(ch for ch in t if not ch.isspace())


def _collapse(t):
    return ' '.join(t.split())


class Stripped:
    """A text with all whitespace removed, with a map back to line numbers."""

    def __init__(self, text):
        self.text = text
        chars, offs = [], []
        for i, ch in enumerate(text):
            if not ch.isspace():
                chars.append(ch)
                offs.append(i)
        self.s = ''.join(chars)
        self.offs = offs
        self.line_starts = [0] + [m.end() for m in re.finditer('\n', text)]

    def line_of(self, char_offset):
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= char_offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def find_all(self, needle):
        n = _strip(needle)
        hits, start = [], 0
        while n:
            i = self.s.find(n, start)
            if i < 0:
                break
            hits.append([self.line_of(self.offs[i]), self.line_of(self.offs[i + len(n) - 1])])
            start = i + 1
        return hits


# ---------------------------------------------------------------------------------------------------------------------
#  A. manuscript identity
# ---------------------------------------------------------------------------------------------------------------------
def check_manuscript():
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex', 'response_to_reviewers_draft.tex', 'cover_letter.tex',
                   'highlights.tex', 'bibliography.bib', 'section2_expert_draft.tex'], cwd=CLONE)
    tex = _read(MAIN_TEX)
    n_lines = tex.count('\n') + (0 if tex.endswith('\n') else 1)
    out = {'clone_head': head, 'clone_head_matches_declared': head == DECLARED_CLONE_HEAD,
           'main_tex_sha256': _sha(MAIN_TEX), 'main_tex_lines': n_lines,
           'letter_sha256': _sha(LETTER), 'cover_letter_sha256': _sha(COVER), 'highlights_sha256': _sha(HIGHLIGHTS),
           'bibliography_sha256': _sha(BIB), 'section2_expert_draft_sha256': _sha(DRAFT2),
           'clone_files_modified': status.strip().splitlines(),
           'last_commit_changed_files': _git(['diff', '--stat', 'c71ca2e', '8b76423'], cwd=CLONE).strip().splitlines()}
    out['main_tex_sha256_matches_declared'] = out['main_tex_sha256'] == DECLARED_MAIN_SHA256
    out['main_tex_lines_match_declared'] = n_lines == DECLARED_MAIN_LINES
    out['letter_sha256_matches_declared_prefix'] = out['letter_sha256'].startswith(DECLARED_LETTER_SHA256_PREFIX)
    for p in (LETTER, COVER, HIGHLIGHTS, BIB, DRAFT2):
        INPUTS.add(p)
    if not (out['clone_head_matches_declared'] and out['main_tex_sha256_matches_declared']
            and out['main_tex_lines_match_declared'] and out['letter_sha256_matches_declared_prefix']
            and not out['clone_files_modified']):
        raise SystemExit(f'manuscript identity check failed: {out}')
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  B. round-4 presence
# ---------------------------------------------------------------------------------------------------------------------
def check_round4():
    md = _read(ROUND4)
    md_c = _strip(md)
    blocks = re.findall(r'```latex\n(.*?)```', md, flags=re.S)
    if len(blocks) != 1 or 'minimum-curtailment term of 1~\\euro/MWh' not in blocks[0]:
        raise SystemExit(f'expected exactly one latex block (item 9) in {ROUND4}; found {len(blocks)}')
    item9_new = blocks[0]
    texts = {'main': Stripped(_read(MAIN_TEX)), 'letter': Stripped(_read(LETTER))}
    out = []
    for item_id, where, new, old in ITEMS:
        if item_id == '9':
            new = item9_new
        rec = {'id': item_id, 'file': where}
        for label, t in (('new', new), ('old', old)):
            if t is None:
                continue
            in_md = _strip(t) in md_c
            if item_id not in EXEMPT_TRANSCRIPTION and not in_md and not (item_id.startswith('6') and label == 'old'):
                raise SystemExit(f'item {item_id} {label} text is not in {ROUND4} (transcription error): {t!r}')
            rec[f'{label}_text'] = _collapse(t)
            rec[f'{label}_text_in_round4_file'] = in_md
        st = texts[where]
        if new is not None:
            rec['new_hits_lines'] = st.find_all(new)
            rec['new_present'] = bool(rec['new_hits_lines'])
        if old is not None:
            rec['old_hits_lines'] = st.find_all(old)
            rec['old_absent'] = not rec['old_hits_lines']
        out.append(rec)
    # item 9: the opening clause kept, the replaced range's old tail absent, the deleted CONFIRM absent
    main = texts['main']
    raw = _read(MAIN_TEX)
    i0 = raw.find('The value of coordination')
    i1 = raw.find('\n\n', i0)
    para = raw[i0:i1]
    item9_extra = {'opening_clause_lines': main.find_all(ITEM9_OPEN),
                   'old_sentence_after_opening_absent': not main.find_all(ITEM9_OPEN + '. Each arrangement'),
                   'paragraph_lines': [raw.count('\n', 0, i0) + 1, raw.count('\n', 0, i1) + 1],
                   'numbers_printed_in_paragraph': re.findall(r'\d[\d,.{}\\]*', para),
                   'consistency_pass_magnitudes_printed': bool(re.search(r'44[.,{]|69[.,{]|234', para))}
    # item 14: placement of k <- k + 1 -- after the \If{...}{\textbf{exit}\;} block, before the \While's closing brace
    lines = _read(MAIN_TEX).splitlines()
    k_lines = [i + 1 for i, ln in enumerate(lines) if ln.strip() == '$k \\gets k + 1$\\;']
    placement = []
    for kl in k_lines:
        prev = [ln.strip() for ln in lines[max(0, kl - 4):kl - 1]]
        nxt = lines[kl].strip() if kl < len(lines) else ''
        placement.append({'line': kl, 'three_lines_before': prev, 'line_after': nxt,
                          'after_exit_block': prev[-2:] == ['\\textbf{exit}\\;', '}'],
                          'before_while_close': nxt == '}'})
    while_lines = [i + 1 for i, ln in enumerate(lines) if '\\While{$k \\le k^{\\max}$}' in ln]
    return {'items': out, 'item9_extra': item9_extra,
            'item14_k_increment': {'lines': k_lines, 'placement': placement, 'while_lines': while_lines}}


# ---------------------------------------------------------------------------------------------------------------------
#  C. literal section-4 references; CONFIRM / AUTHOR comments
# ---------------------------------------------------------------------------------------------------------------------
SEC4_PATTERN = re.compile(r'(?:Sub)?[Ss]ections?(?:~|\s)4(?:\.\d+)?|Sec\.(?:~|\s)?4(?:\.\d+)?')


def check_literals():
    tex = _read(MAIN_TEX)
    lines = tex.splitlines()
    sec_lines = [(i + 1, ln) for i, ln in enumerate(lines) if re.match(r'\\section\{', ln) or ln.startswith('\\appendix')]
    numbered = [(n, ln) for n, ln in sec_lines if ln.startswith('\\section{')]
    # numbering: sections before \appendix are 1, 2, 3, ...
    appendix_line = next((n for n, ln in sec_lines if ln.startswith('\\appendix')), None)
    ranges = {}
    k = 0
    for idx, (n, ln) in enumerate(numbered):
        if appendix_line is not None and n > appendix_line:
            break
        k += 1
        end = numbered[idx + 1][0] - 1 if idx + 1 < len(numbered) else len(lines)
        if appendix_line is not None and end >= appendix_line:
            end = appendix_line - 1
        ranges[str(k)] = {'first_line': n, 'last_line': end, 'heading': ln.strip()}
    occ = []
    for i, ln in enumerate(lines, start=1):
        for m in SEC4_PATTERN.finditer(ln):
            sec = next((s for s, r in ranges.items() if r['first_line'] <= i <= r['last_line']), 'appendix/front')
            occ.append({'line': i, 'section': sec, 'match': m.group(0), 'comment_line': ln.lstrip().startswith('%'),
                        'context': ln[max(0, m.start() - 110): m.end() + 60]})
    in_2_3 = [o for o in occ if o['section'] in ('2', '3')]
    comments = {}
    for label, path in (('main.tex', MAIN_TEX), ('response_to_reviewers_draft.tex', LETTER), ('cover_letter.tex', COVER),
                        ('highlights.tex', HIGHLIGHTS), ('section2_expert_draft.tex (not compiled)', DRAFT2)):
        t = _read(path)
        tl = t.splitlines()
        comments[label] = {'[CONFIRM': t.count('[CONFIRM'), '[AUTHOR]': t.count('[AUTHOR]'),
                           '[AUTHOR:': t.count('[AUTHOR:'),
                           'lines_matching_bracket_confirm_any_case': [
                               {'line': i + 1, 'text': ln.strip()[:200]} for i, ln in enumerate(tl)
                               if re.search(r'\[\s*confirm', ln, flags=re.I)],
                           'lines_matching_bracket_author_any_case': [
                               {'line': i + 1, 'text': ln.strip()[:200]} for i, ln in enumerate(tl)
                               if re.search(r'\[\s*author', ln, flags=re.I)]}
    return {'pattern': SEC4_PATTERN.pattern, 'section_ranges': ranges, 'occurrences_all': occ,
            'occurrences_sections_2_3': in_2_3, 'n_sections_2_3': len(in_2_3), 'comment_counts': comments}


# ---------------------------------------------------------------------------------------------------------------------
#  D. records
# ---------------------------------------------------------------------------------------------------------------------
def check_search():
    out = {}
    for name, d in SEARCH.items():
        res = _read_json(d['results'])
        polls, points = [], []
        for r in res.get('poll_history', []):
            dirs = [c for c in r['candidates'] if c['poll_part'] == 'direction']
            n_eval = sum(1 for c in r['candidates'] if c.get('F_eur') is not None and c.get('disposition') != 'rejected_infeasible')
            polls.append({'poll_index': r['poll_index'], 'delta': r['poll_size_delta'], 'decision': r['decision'],
                          'n_directions': len(dirs),
                          'n_directions_admissible': sum(c['disposition'] != 'rejected_infeasible' for c in dirs),
                          'n_new_evaluations': r['n_new_evaluations'], 'n_points_with_F': n_eval})
            for c in r['candidates']:
                if c.get('resolution_eur') is not None:
                    points.append({'poll_index': r['poll_index'], 'label': c.get('label'), 'disposition': c.get('disposition'),
                                   'bar_sum_eur': c.get('bar_sum_eur'), 'sigma_Q_eur': c.get('sigma_Q_eur'),
                                   'resolution_eur': c.get('resolution_eur'),
                                   'F_inc_minus_F_eur': c.get('F_inc_minus_F_eur'), 'outcome': c.get('outcome')})
        res_values = [p['resolution_eur'] for p in points]
        out[name] = {'git_head_at_run': res['git_head_at_run'], 'termination': res['termination'],
                     'termination_certificate_is_null': res.get('termination_certificate') is None,
                     'STOP_FOR_REVIEW': res.get('STOP_FOR_REVIEW'), 'claim_scope': res.get('claim_scope'),
                     'n_new_evaluations': res.get('n_new_evaluations'), 'budget': d['budget'],
                     'sigma_Q_eur': (res.get('sigma_Q') or {}).get('sigma_Q_eur'),
                     'polls': polls, 'n_points_with_resolution': len(points),
                     'search_threshold_min_eur': min(res_values) if res_values else None,
                     'search_threshold_max_eur': max(res_values) if res_values else None,
                     'points': points}
    return out


def check_thresholds(search):
    d = _read_json(FROZEN_TABLES)
    tau = d['constants']['TAU']
    claims = d['tables']['claims']
    rows = []
    for c in claims:
        rows.append({'claim_id': c['claim_id'], 'form': c.get('form'), 'gross_rule': c.get('gross_rule'),
                     'threshold_or_bar': c.get('gross_threshold_or_bar'), 'ref_status': c.get('ref_status'),
                     'other_status': c.get('other_status')})
    # status strings in the frozen claims read 'cert. ...' (certified) or 'UNCERT. ...' (uncertified)
    def _is_cert(x):
        return str(x).startswith('cert')
    cert = [r['threshold_or_bar'] for r in rows if r['threshold_or_bar'] is not None and _is_cert(r['ref_status'])
            and _is_cert(r['other_status'])]
    unc = [r for r in rows if r['threshold_or_bar'] is not None and not (_is_cert(r['ref_status'])
                                                                          and _is_cert(r['other_status']))]
    l_f2 = [r for r in unc if str(r['claim_id']).startswith('L:')]
    s_min = min(v['search_threshold_min_eur'] for v in search.values() if v['search_threshold_min_eur'] is not None)
    return {'tau': tau, 'three_tau': 3 * tau, 'two_tau': 2 * tau,
            'certified_pair_thresholds_min_max': [min(cert), max(cert)] if cert else None,
            'n_certified_pair_claims': len(cert),
            'uncertified_form_bars_min_max': [min(r['threshold_or_bar'] for r in unc), max(r['threshold_or_bar'] for r in unc)] if unc else None,
            'n_uncertified_form_claims': len(unc),
            'L_claims_bars_min_max': [min(r['threshold_or_bar'] for r in l_f2), max(r['threshold_or_bar'] for r in l_f2)] if l_f2 else None,
            'n_L_claims': len(l_f2),
            'search_threshold_min_over_all_points_eur': s_min,
            'search_coarser_than_every_certified_threshold': bool(cert) and s_min > max(cert) and s_min > 3 * tau,
            'search_coarser_than_every_uncertified_bar': bool(unc) and all(s_min > r['threshold_or_bar'] for r in unc),
            's53_threshold_vs_L_bars': {'s53_min': search['s53']['search_threshold_min_eur'],
                                        's53_max': search['s53']['search_threshold_max_eur']},
            'claims': rows}


def check_configuration_identity():
    """Every campaign that produced a certified cell of the frozen tables: code identity vs HEAD, tail and AA declared."""
    d = _read_json(FROZEN_TABLES)
    t = d['tables']
    cells = {}
    for grp in ('cells', 'cells_appended_w160'):
        for k, v in t[grp].items():
            cells[k] = v
    for r in t['phase_b']:
        cells.setdefault(r['cell'], r)
    tracked = _git(['ls-files', 'data/SRP1/Results']).splitlines()
    evrec = [f for f in tracked if f.endswith('evaluation_record.json')]
    pathspec = list(PRODUCTION_FILES) + list(CASE_DATA)
    per_cell, heads_done = {}, {}
    for name, v in sorted(cells.items()):
        if v.get('status') != 'certified':
            continue
        ek = v.get('eval_key')
        recs = [f for f in evrec if ek and f'/{ek[:16]}_' in f]
        if len(recs) != 1:
            raise SystemExit(f'cell {name}: expected one evaluation record for {ek}, found {recs}')
        camp = recs[0].split('/evals/')[0]
        res_path = os.path.join(camp, 'campaign_results.json')
        res = _read_json(res_path)
        heads = sorted(set(_walk(res, 'git_head_at_run')))
        specs = sorted(glob.glob(os.path.join(camp, 'campaign_spec_*.json')))
        cfg = (_read_json(specs[0]).get('configuration') or {}) if len(specs) == 1 else {}
        for h in heads:
            if h not in heads_done:
                stat = _git(['diff', '--stat', h, 'HEAD', '--'] + pathspec).strip().splitlines()
                heads_done[h] = stat[-1] if stat else 'NO CHANGE'
        per_cell[name] = {'campaign': camp, 'git_heads_at_run': heads, 'code_vs_HEAD': [heads_done[h] for h in heads],
                          'spec': specs, 'tail': cfg.get('convergence_depth_tail'),
                          'aa': cfg.get('case_file_anderson_acceleration')}
    control = _git(['diff', '--stat', CONTROL_HEAD, 'HEAD', '--'] + pathspec).strip().splitlines()
    search = {}
    for name, dd in SEARCH.items():
        spec = _read_json(dd['spec'])
        head = _read_json(dd['results'])['git_head_at_run']
        counts = {}
        for f, pat in (('shared_resources_planning.py', 'def _apply_convergence_depth_tail'),
                       ('admm_parameters.py', 'convergence_depth_tail'),
                       ('p515_s44_campaign_harness.py', 'convergence_depth_tail')):
            counts[f] = _git(['show', f'{head}:{f}']).count(pat)
        stat = _git(['diff', '--stat', head, 'HEAD', '--'] + pathspec).strip().splitlines()
        search[name] = {'git_head_at_run': head, 'code_vs_HEAD': stat[-1] if stat else 'NO CHANGE',
                        'spec_configuration_declares_tail': 'convergence_depth_tail' in (spec.get('configuration') or {}),
                        'tail_code_occurrences_at_head': counts}
    n_cert = len(per_cell)
    return {'n_certified_cells': n_cert,
            'n_cells_code_identical_to_HEAD': sum(all(s == 'NO CHANGE' for s in v['code_vs_HEAD']) for v in per_cell.values()),
            'n_cells_tail_enabled': sum(bool((v['tail'] or {}).get('enabled')) for v in per_cell.values()),
            'n_cells_aa_enabled_memory5': sum(bool((v['aa'] or {}).get('enabled')) and (v['aa'] or {}).get('memory') == 5
                                              for v in per_cell.values()),
            'campaign_roots': sorted({v['campaign'].split('/campaign_')[0] for v in per_cell.values()}),
            'control': {'head': CONTROL_HEAD, 'last_line': control[-1] if control else 'NO CHANGE'},
            'per_cell': per_cell, 'search_campaigns': search}


def check_c4():
    spec = _read_json(EXT_SPEC_V3)
    arm = spec['model_variant']['arms']['C4']
    cell = spec['cells']['e_c4']
    base = cell['configuration']['ess_ageing_baseline']['calibration']
    n, dod, r = base['cycles_n'], base['reference_dod_d'], arm['eol_retention_r']
    k_entered = n * dod / (-math.log(r))
    k_table = 8000 * 1.00 / (-math.log(0.70))
    return {'arm': arm, 'base_calibration': base, 'entered_triple': [n, dod, r], 'product_entered': n * dod,
            'product_table_row': 8000 * 1.00, 'k_entered': k_entered, 'k_table_row_triple': k_table,
            'k_closed_form_spec': spec['model_variant']['k_closed_form_by_arm']['C4'],
            'k_equal': abs(k_entered - k_table) < 1e-9 and abs(k_entered - spec['model_variant']['k_closed_form_by_arm']['C4']) < 1e-6,
            'k_printed': 22429, 'k_rounds_to_printed': round(k_entered) == 22429}


def check_benchmark():
    d = _read_json(FROZEN_TABLES)['tables']['benchmark']
    runs = {}
    for p in NRF_RUNS:
        r = _read_json(p)
        blocks = r['phase_B_consistency']['reevaluation']['blocks']
        trig = [k for k, v in blocks.items() if v['violations']['trigger_sequential_pass']]
        dns = sorted({k.split('|')[1] for k in blocks})
        dns_trig = sorted({k.split('|')[1] for k in trig})
        runs[os.path.basename(p)] = {
            'n_dso_blocks': len(blocks), 'n_blocks_triggering': len(trig), 'dns': dns, 'dns_with_a_trigger': dns_trig,
            'global_trigger': r['phase_B_consistency']['reevaluation']['trigger_sequential_pass'],
            'arm_cost_source': r['arm_cost']['source'], 'arm_cost': r['arm_cost']['gross_operational_cost'],
            'effect_on_q_eur': (r.get('phase_C_sequential_pass') or {}).get('effect_on_q_eur'),
            'decision_tie_breaker': r.get('decision_tie_breaker'), 'evaluation_tie_breaker': r.get('evaluation_tie_breaker')}
    INPUTS.add(BENCH_SPEC_V5)
    return {'instance': d['instance'], 'frozen_spec': d['frozen_spec'], 'arms': d['arms'], 'benefit': d['benefit'],
            'benefit_relative': d['benefit_relative'], 'runs': runs,
            'every_dn_block_triggered_in_every_run': all(v['n_blocks_triggering'] == v['n_dso_blocks'] == 36
                                                         for v in runs.values()),
            'every_reported_value_after_pass': all(v['arm_cost_source'] == 'phase_C_sequential_pass' for v in runs.values()),
            'bench_spec_v5_sha256': _sha(BENCH_SPEC_V5)}


def check_chemistry():
    pat = re.compile(r'lfp|lithium|phosphate|lifepo|li-ion|li ion|nmc|chemistr|mb31', re.I)
    out = {}
    for p in ESS_FILES:
        INPUTS.add(p)
        if p.endswith('.xlsx'):
            import openpyxl  # read-only, no project code
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            hits, n_cells = [], 0
            for ws in wb.worksheets:
                for row in ws.iter_rows():
                    for c in row:
                        if c.value is None:
                            continue
                        n_cells += 1
                        if isinstance(c.value, str) and pat.search(c.value):
                            hits.append({'sheet': ws.title, 'cell': getattr(c, 'coordinate', None), 'value': c.value})
            wb.close()
            out[p] = {'sha256': _sha(p), 'n_nonempty_cells': n_cells, 'hits': hits}
        else:
            t = _read(p)
            out[p] = {'sha256': _sha(p), 'hits': [m.group(0) for m in pat.finditer(t)]}
    bib = _read(BIB)
    i = bib.find('@misc{nrel_ess_costs,')
    out['bibliography nrel_ess_costs'] = bib[i: bib.find('}\n\n', i) + 1] if i >= 0 else None
    return out


def check_pyomo_sol():
    spec = importlib.util.find_spec('pyomo.opt.plugins.sol')
    path = spec.origin
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read().splitlines()
    # the AMPL solve_result_num is read into objno[1]; 0-99 = solved, 100-199 = solved?
    hits = [{'line': i + 1, 'text': ln.strip()} for i, ln in enumerate(src) if 'objno[1] >=' in ln]
    ctx = []
    for h in hits[:2]:
        ctx.append([s.strip() for s in src[h['line'] - 1: h['line'] + 4]])
    import pyomo
    return {'path': path, 'sha256': _sha(path), 'pyomo_version': pyomo.version.version, 'hits': hits, 'context': ctx}


def check_anchors():
    out = {}
    for anchor_id, path, text in CODE_ANCHORS:
        src = _read(path)
        hits = [i for i, line in enumerate(src.splitlines(), start=1) if text in line]
        if not hits:
            raise SystemExit(f'anchor {anchor_id} not found in {path}: {text!r}')
        out[anchor_id] = {'file': path, 'text': text, 'lines': hits, 'file_sha256': _sha(path)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'refusing to overwrite {OUT_JSON} / {OUT_MANIFEST}')
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manuscript = check_manuscript()
    search = check_search()
    for p in (FROZEN_TABLES, EXT_SPEC_V3) + tuple(NRF_RUNS) + tuple(d['spec'] for d in SEARCH.values()) \
            + tuple(d['results'] for d in SEARCH.values()):
        INPUTS.add(p)
    result = {
        'schema': 'p515_s53_w176a_round4_lines_checks_v1',
        'stage': 'P5.15 Addendum 72 W176a -- round-4 lines at Overleaf 8b76423: presence, records, literal section-4 '
                 'references, CONFIRM/AUTHOR comments',
        'started_utc': started,
        'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'interpreter': sys.executable,
        'A_manuscript': manuscript,
        'B_round4': check_round4(),
        'C_literals_and_comments': check_literals(),
        'D1_search': search,
        'D2_thresholds': check_thresholds(search),
        'D3_configuration_identity': check_configuration_identity(),
        'D4_c4': check_c4(),
        'D5_benchmark': check_benchmark(),
        'D6_chemistry': check_chemistry(),
        'D7_pyomo_sol_mapping': check_pyomo_sol(),
        'E_code_anchors': check_anchors(),
    }
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    if bad or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: production modules {bad}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                        'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, sort_keys=False)
        f.write('\n')
    script = os.path.basename(__file__)
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'script': {script: _sha(script)},
                'inputs': {p: _sha(p) for p in sorted(INPUTS)}}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write('\n')

    # ------------------------------------------------------------------ summary
    print(f"W176a: clone {manuscript['clone_head']} main.tex {manuscript['main_tex_sha256'][:8]} "
          f"lines {manuscript['main_tex_lines']} letter {manuscript['letter_sha256'][:8]}; last commit "
          f"{manuscript['last_commit_changed_files']}")
    for it in result['B_round4']['items']:
        print(f"  item {it['id']}: new_present={it.get('new_present')} lines={it.get('new_hits_lines')} "
              f"old_absent={it.get('old_absent')} old_lines={it.get('old_hits_lines')}")
    print(f"  item 9 extra: {result['B_round4']['item9_extra']}")
    print(f"  item 14 k+1: {json.dumps(result['B_round4']['item14_k_increment'])}")
    c = result['C_literals_and_comments']
    print(f"(c) section ranges: {json.dumps(c['section_ranges'])}")
    print(f"(c) literal section-4 refs in sec. 2-3: {c['n_sections_2_3']}")
    for o in c['occurrences_all']:
        print(f"    l. {o['line']} sec {o['section']} [{o['match']}] comment={o['comment_line']} :: {o['context']}")
    print(f"(c) comment counts: {json.dumps(c['comment_counts'])}")
    for n, v in search.items():
        print(f"search {n}: {v['termination']['reason']} certificate_null={v['termination_certificate_is_null']} "
              f"new {v['n_new_evaluations']}/{v['budget']} admissible per poll "
              f"{[(p['poll_index'], p['n_directions_admissible'], p['decision']) for p in v['polls']]} "
              f"threshold min/max {v['search_threshold_min_eur']}/{v['search_threshold_max_eur']}")
    th = result['D2_thresholds']
    print(f"thresholds: 3tau {th['three_tau']:.2f}; certified-pair claims {th['n_certified_pair_claims']} "
          f"{th['certified_pair_thresholds_min_max']}; uncertified-form {th['n_uncertified_form_claims']} "
          f"{th['uncertified_form_bars_min_max']}; L claims {th['n_L_claims']} {th['L_claims_bars_min_max']}; "
          f"search min {th['search_threshold_min_over_all_points_eur']}; coarser than certified "
          f"{th['search_coarser_than_every_certified_threshold']}, than uncertified {th['search_coarser_than_every_uncertified_bar']}")
    ci = result['D3_configuration_identity']
    print(f"config identity: certified cells {ci['n_certified_cells']}, code==HEAD {ci['n_cells_code_identical_to_HEAD']}, "
          f"tail on {ci['n_cells_tail_enabled']}, AA {ci['n_cells_aa_enabled_memory5']}; roots {ci['campaign_roots']}; "
          f"control {ci['control']}; search {json.dumps(ci['search_campaigns'])}")
    print(f"C4: {json.dumps({k: v for k, v in result['D4_c4'].items() if k not in ('arm', 'base_calibration')})}")
    b = result['D5_benchmark']
    print(f"benchmark: instance {b['instance']['label']} {b['instance']['problem']}; every DN block triggered "
          f"{b['every_dn_block_triggered_in_every_run']}; every value after pass {b['every_reported_value_after_pass']}")
    print(f"chemistry: {json.dumps({k: (v.get('hits') if isinstance(v, dict) else v) for k, v in result['D6_chemistry'].items()})}")
    print(f"pyomo sol: {json.dumps(result['D7_pyomo_sol_mapping']['context'])}")
    print(f"anchors: {json.dumps({k: v['lines'] for k, v in result['E_code_anchors'].items()})}")
    print(f"guards: {result['guards']}")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
