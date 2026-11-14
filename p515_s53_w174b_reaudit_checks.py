"""
P5.15 Addendum 70 order, Planner task W174b -- static checks behind the re-audit of the W171b rows (section 2) and the
W172 rows (Appendix A) at Overleaf 260bd83 (P5_15_W174B_REAUDIT.md). ZERO SOLVES, NO MODEL, NO PICKLE.

What it does (static reading only):
  A. Manuscript identity: the Overleaf clone's HEAD (declared 260bd83), main.tex sha256 (declared, full hash), line
     count (2,209), main.tex / bibliography.bib unmodified in the clone.
  B. Correction-block presence: every replacement block of STEP6_ROUND2_CORRECTIONS.md (A.1-A.4, B.1-B.5, C.1-C.8) and
     STEP6_ROUND3_CORRECTIONS.md (A.1, A.2 a-c, A.3 d-s, B a-d, C.1-C.3, and the section-reference rule of C) searched in
     main.tex with ALL whitespace removed. The fenced ```latex blocks are parsed from the two files (by order, the order
     asserted by a content check); the inline find -> replace strings are transcribed below verbatim from the files.
     Where round 3 rewrote text that round 2 introduced (Algorithm 1 and the paragraph after it), the expected text is
     the round-2 block with the round-3 replacements applied. Status: present_verbatim / present_with_changes (word
     diff against the best-matching region) / absent; old text checked absent where the block names one.
  C. The `% [CONFIRM -- W17x]` comments still in main.tex, with line numbers.
  D. Code identity: `git diff --stat <git_head_at_run> HEAD -- <20 production files + SRP1 case data>` for every
     campaign_results.json behind the tables (as W172), one subprocess per head, argument list (no shell); positive
     control 353e094b must report changes. Informational: the same diff for the three master-search heads.
  E. Record checks:
     E1 the poll histories of the three search campaigns (direction dispositions per poll, termination reason, new
        evaluations used / budget);
     E2 the tight tail in the search campaigns: `configuration.convergence_depth_tail` declared in their specs? does
        the tail exist in the code at their git heads (`git show <head>:<file>`, grep counts)?
     E3 the certifying spec of every frozen-table cell (frozen_step6_tables_v1_590088fe.json) and the v6-from-records
        decision of each committed pre-v6 record (w142 v6_from_records) against its committed k*; the W168 x0 replay.
  F. Code anchors: exact-text search of each code location the report cites, line numbers at HEAD (every anchor must
     be found).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit; pyomo is imported by the guard only). Refuses to overwrite an existing output.

Output (new directory): data/SRP1/Results/P515S53/w174b_reaudit/w174b_checks.json and manifest_sha256.json.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S53/w174b_reaudit && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w174b_reaudit_checks.py \\
      > data/SRP1/Results/P515S53/w174b_reaudit/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W174b re-audit checks (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W174b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W174b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import difflib  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w174b_reaudit')
OUT_JSON = os.path.join(OUT_DIR, 'w174b_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
DECLARED_CLONE_HEAD = '260bd83'
DECLARED_MAIN_SHA256 = 'cb237b2d129e689a94dd50abec9a9cca739567ef97a486ec012a901f0e04b6df'
DECLARED_MAIN_LINES = 2209
ROUND2 = 'STEP6_ROUND2_CORRECTIONS.md'
ROUND3 = 'STEP6_ROUND3_CORRECTIONS.md'

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
CAMPAIGN_ROOTS = (os.path.join(P53, 'w142_resettle_v6'), os.path.join(P53, 'w142_resettle_ext_v6'),
                  os.path.join(P53, 'w155_a64_cells'), os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'))
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
FROZEN_TABLES = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
V6_FROM_RECORDS = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json')
W168_DECISION = os.path.join(P53, 'w168_x0_v6_replay', 'w168_x0_v6_replay_decision.json')
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

# ---------------------------------------------------------------------------------------------------------------------
#  inline find -> replace items, transcribed verbatim from the two correction files
# ---------------------------------------------------------------------------------------------------------------------
R2_INLINE = [
    ('R2-A.1(c)', 'By contrast, the network dispatch $\\boldsymbol{u}_o$ adapts to each realization of load, RES generation and '
                  'market prices, while the day-ahead decisions $\\boldsymbol{u}^{0}$ do not: the recourse of each '
                  'representative day is itself a two-stage problem (Subsection~\\ref{subsubsec:commitment}).', None),
    ('R2-A.4', 'it uses the value $Q(\\boldsymbol{x})$ returned by the recourse and the record of its evaluation (its bar and '
               'exit), nothing else', 'it uses the value $Q(\\boldsymbol{x})$ returned by the recourse and nothing else'),
    ('R2-B.2(item 3)', '$n_{\\mathrm{w}} = \\max\\{20, \\lceil 1.1\\,\\hat{P} \\rceil\\}$ cycles, all after $k_0$,', None),
    ('R2-B.4(a)', 'A difference involving uncertified evaluations is determinate only if it exceeds three times the largest '
                  'of their consensus gaps and settling slacks, in both gross and gap-corrected terms (the objective plus '
                  'the priced gap).',
     'A difference involving an uncertified evaluation is determinate only if it exceeds three times the larger of that '
     'evaluation\'s consensus gap and settling slack.'),
    ('R2-B.4(b)', 'when it is at least $\\max\\{3\\,b, 2\\tau\\}$', 'when it exceeds $\\max\\{3\\,b, 2\\tau\\}$'),
    ('R2-B.5(a)', 'generation cost in the TN at the scenario\'s market price, activated flexibility in the DNs, load '
                  'curtailment at its price, the closure-slack penalty of the storage model '
                  '(Subsection~\\ref{subsubsec:network_storage}) and the feasibility-slack penalties of the network '
                  'models, all at their solver bound in every reported evaluation',
     'generation cost in the TN, activated flexibility and curtailment in the DNs, the closure-slack penalty'),
    ('R2-B.5(b)', 'Renewable curtailment carries no cost in $Q(\\boldsymbol{x})$; it is reported as energy.', None),
    ('R2-C.2(a)', 'The net active power of the storage, in the consumption convention shared with the agent, is '
                  '$P^{\\text{E}}_{e,y,d,t} = P^{\\text{Ch}}_{e,y,d,t} - P^{\\text{Dch}}_{e,y,d,t}$ (positive when charging)',
     '(positive when discharging)'),
    ('R2-C.2(b)', 'so that the net schedule the agent ages is the one every scenario runs', None),
    # inside eq. (soc_closure) the member carries no $ delimiters; the block's own $...$ are dropped for the search
    ('R2-C.3(a)', 'E^{\\text{SoC}}_{e,y,d,0} = SoC^{0}_e \\, E^{\\text{Av}}_{e,y}', 'E^{\\text{SoC}}_{e,y,d,t_0}'),
    ('R2-C.3(b)', 'and the stored energy follows, from the constant pre-day state $E^{\\text{SoC}}_{e,y,d,0}$', None),
    ('R2-C.3(c)', 'the slack is penalised at $c^{\\text{Cl}}$ per MWh in the objective of each network model that carries '
                  'the storage', 'per MWh in the block objective'),
    ('R2-C.3(d)', 'The closure slack was at its lower bound, to solver tolerance, at every certificate used in this paper.',
     '[CONFIRM — W169]'),
    ('R2-C.4(a)', 'up to a slack pair that is penalised in its objective and was at its lower bound at every certified point',
     'is never active at a consensus point within the agent\'s ratings'),
    ('R2-C.4(b)', 'and the converter circle are the agent\'s nonlinear rows', None),
    ('R2-C.5(a)', 'It is recorded, not enforced; at the certified points it was below $4 \\times 10^{-5}$ of the rating. '
                  'The slack pair $\\sigma$ was at its lower bound at every certified point.', None),
    ('R2-C.5(b)', 'is checked after every solve and recorded', None),
    # C.5 (a) and (b) as merged in the paste (one sentence instead of 'recorded. It is recorded, not enforced; ...')
    ('R2-C.5(a+b merged)', 'is checked after every solve and recorded, not enforced; at the certified points it was below '
                           '$4 \\times 10^{-5}$ of the rating. The slack pair $\\sigma$ was at its lower bound at every '
                           'certified point.', None),
    ('R2-C.6(a)', 'In the multi-scenario instance every scenario $s = (m, o)$ --- a market scenario $m \\in \\Omega_M$ and an '
                  'operation scenario $o \\in \\Omega_O$ of load and RES generation, with probability $\\omega_s = '
                  '\\omega_m \\omega_o$ --- is held inside every network block, while',
     'In the multi-scenario instance each operation scenario $o$ is instantiated inside every network block'),
    ('R2-C.6(b)', '($\\forall i$, $s$, $t$)', '($\\forall i$, $o$, $t$)'),
    ('R2-C.6(c)', 'With a single scenario \\eqref{eq:commitment_deviation}--\\eqref{eq:commitment_charge} vanish '
                  'identically, so the single-scenario instance is unaffected by them; the commitment is to the '
                  'expectation over market and operation scenarios alike, so deviations driven by the price realization '
                  'are charged like those driven by load and generation.',
     'At a single operation scenario'),
    ('R2-C.7', 'a dedicated shared-ESS subproblem per interface node, spanning all representative years and days', None),
    ('R2-C.8', '22\\,429', '22{,}430'),
]
R3_INLINE = [
    ('R3-A.1', 'At a single scenario the commitment charge and the voltage regularisation are not constructed; the '
               'interface settlement remains in every local objective at full weight and is excluded from the reported '
               'cost as a transfer.', 'At a single scenario all of these terms vanish identically.'),
    ('R3-A.2(a)', 'in the multi-scenario instance these are the \\emph{expected} interface quantities, $\\sum_{s \\in '
                  '\\Omega_M \\times \\Omega_O} \\omega_s P^{\\text{I}}_{i,s,t}$, of Subsection~\\ref{subsubsec:commitment}',
     '$\\sum_{o} \\omega_o P^{\\text{I}}_{i,o,t}$'),
    ('R3-A.2(b)', 'which is the minimiser over $z$ of the three agents\' consensus terms taken with unit weight; the factor '
                  '$\\kappa^{\\text{E}}$ scales the storage agent\'s local problem only and does not enter the update.',
     'which is the minimiser of the sum of the three agents\' storage terms over $z$.'),
    ('R3-A.3(d)', 'run as a sweep (DSOs, then TSO, then the storage agent) that is Gauss--Seidel on the interface channels '
                  'and, on the storage channel, a global-variable consensus in which all three agents solve against the '
                  'same $z$ before it is updated', 'run as a Gauss--Seidel sweep'),
    ('R3-A.3(e)', 'so that every block is solved in the same units and the sum over blocks, after the exclusions of '
                  'Subsection~\\ref{subsubsec:certification}, is the discounted operating cost of '
                  '\\eqref{eq:recourse_value}.', None),
    ('R3-A.3(f)', 'which puts the agent\'s local objective on the same footing as a median block\'s scaled objective; the '
                  'consensus terms themselves are unscaled in every agent.',
     'which puts them in the same units as the network blocks\' scaled objectives.'),
    ('R3-A.3(g1)', '\\kappa^{\\text{E}} \\left( \\mathcal{L}^{E,P}_{e,y,d,t} + \\mathcal{L}^{E,Q}_{e,y,d,t} \\right)', None),
    ('R3-A.3(g2)', 'Here $\\mathcal{L}^{E,P}$ and $\\mathcal{L}^{E,Q}$ are the agent\'s consensus terms on its net active and '
                   'reactive power, written out in \\eqref{eq:admm_esso_local}, and $\\kappa^{\\text{E}}$ the scale of '
                   '\\ref{app:admm_updated_agents}.', None),
    ('R3-A.3(h)', 'An agent whose solve failed keeps its previous copy; the interface duals of a block are updated only when '
                  'both its TSO and its DSO solves succeeded, and $z$ and the three storage duals of a node only when all '
                  'three solves succeeded.', 'No consensus or dual update is made for a block or node'),
    ('R3-A.3(i)', 'all $\\rho_g$ are frozen from the cycle after the first residual pass, or from the stopping cycle of the '
                  'earlier run in an evaluation continued from one.', None),
    ('R3-A.3(j1)', '$s_{\\text{E}} = \\lVert \\rho^{\\text{E}} \\alpha\\, \\Delta z \\rVert$ taken over the three agents (so '
                   '$\\sqrt{3}\\,\\rho^{\\text{E}} \\alpha \\lVert \\Delta z \\rVert$)', None),
    ('R3-A.3(j2)', '$\\varepsilon^{\\text{dual}} = \\sqrt{n}\\,\\varepsilon_{\\text{abs}} + \\varepsilon_{\\text{rel}} \\lVert '
                   '\\lambda \\rVert$, with $\\lambda$ the DSO-side dual on the interface channels and all three duals on '
                   'the storage channel', None),
    ('R3-A.3(k)', 'The single-scenario evaluations of this paper were stopped by the certification rule of '
                  'Subsection~\\ref{subsubsec:certification} or by its cycle cap, whichever came first; the holds start '
                  'at the first passing cycle, while the rule\'s own $k_0$ resets on a lapse.',
     'which uses the first passing cycle $k_0$ as its starting point.'),
    ('R3-A.3(l)', 'is applied, on the storage channel, to $z$ and the three agents\' $\\lambda_a/\\rho$ and, on each interface '
                  'channel, to the TSO\'s copy and the DSO-side $\\lambda/\\rho$', 'is applied to the pair $(z, \\lambda/\\rho)$ of every channel'),
    ('R3-A.3(m1)', 'from the cycle after $k_0$ on (or after the earlier run\'s stopping cycle in a continued evaluation)', None),
    ('R3-A.3(m2)', 'the certifying regime holds the tail on from the cycle after $k_0$', None),
    ('R3-A.3(n)', 'the production setting restores each operator\'s own tolerance otherwise (the tail is a declared option, '
                  'enabled in every campaign)', 'restores the default tolerance otherwise'),
    ('R3-A.3(o)', 'A local solve that ends at the iteration limit, infeasible or in a solver error is retried in two tiers, a '
                  'cold restart with the agent\'s recovery settings and the same with the adaptive barrier strategy '
                  '(Section~\\ref{sec:case_settings}); an exit at the solver\'s acceptable level is counted as a successful '
                  'solve for the stopping test, and the certification rule decides separately whether the cycle is clean.',
     'documented sequence of solver settings'),
    ('R3-A.3(p)', 'after the initialisation solves, the plan $\\boldsymbol{x}$ enters a network model only through these '
                  'capacities.', 'itself never enters a network model except through these'),
    ('R3-A.3(r)', 'record the objective, residuals and solve statuses of the cycle (the priced interface gap is recorded by '
                  'the campaign harness)', 'residuals, gap and solve'),
    ('R3-A.3(s)', 'the TSO represents each DN by the DN\'s expected exchange at initialisation, held fixed, plus a '
                  'scenario-free adjustment bounded by the interface rating as its only interface freedom (the same '
                  'construction at one scenario)', 'the TSO represents each DN by that expected exchange with'),
    ('R3-B.a(1)', '$\\Delta \\gets \\Delta_0$ ($\\Delta_0 = 4$ lattice units in variant A; $\\Delta = 1$ throughout in variant B)',
     None),
    ('R3-B.a(2)', 'initial poll size $\\Delta_0$;', None),
    ('R3-B.b', '\\While{polls remain (at most 60) and a poll can be launched}{', '\\While{$N < N^{\\max}$}{'),
    ('R3-B.c', 'if fewer than $n+1$ distinct admissible points result, add every admissible unit neighbour (the poll is '
               'refused for review beyond 30 points)\\;', 'add admissible unit neighbours until $n+1$'),
    ('R3-B.d(1)', 'at the plan found under the doubled flexibility price, seventeen of its 61 admissible unit neighbours were '
                  'evaluated (the ten points of the final poll and seven earlier evaluations) and none is determinately '
                  'better; of the thirteen re-evaluated under the certification rule the plan is better than twelve, seven '
                  'determinately and five within resolution, and within resolution of the thirteenth --- it is not '
                  'reported as a mesh-local optimum.', 'the final poll evaluated twelve neighbours'),
    ('R3-B.d(2)', 'in which every rounded poll direction was inadmissible at every poll, so that every evaluated point came '
                  'from the unit-neighbour completion,', None),
    ('R3-C.2', 'Conventional generation in the TN is priced at the wholesale energy price of the scenario; no separate '
               'generator cost curves are used.', None),
    ('R3-C.3(a)', None, '\\textcolor{red}'),
    ('R3-C.3(b)', 'Branch 1 is the interface transformer, rated 200, 100 and 150~MVA for the ADNs at TN nodes 5, 7 and 9', None),
    ('R3-C(refs a)', 'are given in Section~\\ref{sec:case_ess_params}.', 'are given in Section~3.4.'),
    ('R3-C(refs b)', None, 'Section~3.5'),
]
# Text whose deletion by the round-3 paste is NOT named by any correction block (adjacent-text checks).
ADJACENT = [
    ('R3-A.2(c) k <- k+1', '$k \\gets k + 1$\\;'),
    ('R3-A.3(m) AA off clause', 'and the acceleration is switched off on any cycle in which every channel passes'),
    ('R3-A.3(d) join', 'before it is updated with residual balancing of the penalty parameters'),
]


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


INPUTS = set()


def _read(path):
    INPUTS.add(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout


def _walk(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


# ---------------------------------------------------------------------------------------------------------------------
#  A. manuscript identity
# ---------------------------------------------------------------------------------------------------------------------
def check_manuscript():
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex', 'bibliography.bib'], cwd=CLONE)
    tex = _read(MAIN_TEX)
    n_lines = tex.count('\n') + (0 if tex.endswith('\n') else 1)
    sha = _sha(MAIN_TEX)
    INPUTS.add(BIB)
    out = {'clone_head': head, 'clone_head_matches_declared': head == DECLARED_CLONE_HEAD,
           'main_tex_sha256': sha, 'main_tex_sha256_matches_declared': sha == DECLARED_MAIN_SHA256,
           'main_tex_lines': n_lines, 'main_tex_lines_match_declared': n_lines == DECLARED_MAIN_LINES,
           'main_or_bib_modified_in_clone': bool(status.strip()), 'bibliography_sha256': _sha(BIB),
           'walker_ni_2011_entries': _read(BIB).count('{walker_ni_2011,')}
    if not (out['clone_head_matches_declared'] and out['main_tex_sha256_matches_declared']
            and out['main_tex_lines_match_declared'] and not out['main_or_bib_modified_in_clone']):
        raise SystemExit(f'manuscript identity check failed: {out}')
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  B. correction-block presence
# ---------------------------------------------------------------------------------------------------------------------
class Stripped:
    """main.tex with all whitespace removed, with a map back to (char offset, line number)."""

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
            hits.append((self.line_of(self.offs[i]), self.line_of(self.offs[i + len(n) - 1])))
            start = i + 1
        return hits

    def region_text(self, i0, i1):
        return self.text[self.offs[i0]: self.offs[i1] + 1]


def _strip(t):
    return ''.join(ch for ch in t if not ch.isspace())


def _latex_blocks(md):
    return re.findall(r'```latex\n(.*?)```', md, flags=re.S)


def _word_diff(expected, found):
    a, b = expected.split(), found.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [{'op': op, 'expected': ' '.join(a[i1:i2]), 'found': ' '.join(b[j1:j2])}
            for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']


def _best_region(st, expected, anchor_len=40):
    """Locate the region of main.tex that best matches `expected` (first and last `anchor_len` stripped characters)."""
    e = _strip(expected)
    head, tail = e[:anchor_len], e[-anchor_len:]
    i = st.s.find(head)
    if i < 0:
        return None
    j = st.s.find(tail, i)
    if j < 0 or j - i > 3 * len(e):
        # fall back: the stretch of the same length after the head anchor
        j = min(len(st.s) - anchor_len, i + len(e))
    return i, j + len(tail) - 1


def block_status(st, item_id, expected, old=None):
    rec = {'id': item_id}
    if expected is not None:
        hits = st.find_all(expected)
        rec['lines'] = hits
        if hits:
            rec['status'] = 'present_verbatim'
        else:
            reg = _best_region(st, expected)
            if reg is None:
                rec['status'] = 'absent'
            else:
                found = st.region_text(*reg)
                rec.update({'status': 'present_with_changes', 'region_lines': [st.line_of(st.offs[reg[0]]),
                                                                                st.line_of(st.offs[reg[1]])],
                            'word_diff': _word_diff(expected, found)})
    if old is not None:
        old_hits = st.find_all(old)
        rec['old_text_absent'] = not old_hits
        rec['old_text_lines'] = old_hits
        if expected is None:
            rec['status'] = 'old_text_absent' if not old_hits else 'old_text_still_present'
    return rec


def _apply(text, find, repl):
    if find not in text:
        raise SystemExit(f'expected-text construction: {find!r} not in the round-2 block')
    return text.replace(find, repl, 1)


def check_blocks(st):
    r2, r3 = _read(ROUND2), _read(ROUND3)
    b2, b3 = _latex_blocks(r2), _latex_blocks(r3)
    # the order of the fenced blocks, asserted by content
    want2 = [('R2-A.1(a) subequations', '\\begin{subequations}'), ('R2-A.1(b) Here items', '\\item $\\boldsymbol{x}$'),
             ('R2-A.2 Algorithm 1', '\\begin{algorithm}'), ('R2-A.3 paragraph', 'is a mesh adaptive direct search'),
             ('R2-B.1', 'The residual pass at cycle $k_0$'), ('R2-B.2 items 1-2', 'turning points since $k_0$'),
             ('R2-B.3', 'An objective that descends without a sign change'), ('R2-C.1', 'The net storage schedule')]
    want3 = [('R3-A.2(c)', 'Apply the Anderson step'), ('R3-A.3(q)', 'Convert every model to its ADMM form'),
             ('R3-C.1 sentence', 'The three trajectories enter'), ('R3-C.2 block', 'Conventional generation in the TN'),
             ('R3-C.4 section 3.5', 'Shared Energy Storage Parameters'), ('R3-C.5 section 3.6', 'Evaluation and Certification')]
    if len(b2) != len(want2) or len(b3) != len(want3):
        raise SystemExit(f'latex block counts {len(b2)} / {len(b3)} != {len(want2)} / {len(want3)}')
    for blocks, want in ((b2, want2), (b3, want3)):
        for blk, (bid, marker) in zip(blocks, want):
            if marker not in blk:
                raise SystemExit(f'latex block order: {bid} lacks marker {marker!r}')
    out = []
    # round 2: fenced blocks (A.2 and A.3 checked as-is and with the round-3 B replacements applied)
    alg_final = b2[2]
    alg_final = _apply(alg_final, '$\\Delta \\gets \\Delta_0$;',
                       '$\\Delta \\gets \\Delta_0$ ($\\Delta_0 = 4$ lattice units in variant A; $\\Delta = 1$ throughout in '
                       'variant B);')
    alg_final = _apply(alg_final, 'scenarios and probabilities;', 'scenarios and probabilities; initial poll size $\\Delta_0$;')
    alg_final = _apply(alg_final, '\\While{$N < N^{\\max}$}{', '\\While{polls remain (at most 60) and a poll can be launched}{')
    alg_final = _apply(alg_final, 'add admissible unit neighbours until $n+1$\\;',
                       'add every admissible unit neighbour (the poll is refused for review beyond 30 points)\\;')
    # the round-2 paragraph is wrapped over lines in the md file; apply B.d on the whitespace-normalised form
    par_norm = ' '.join(b2[3].split())
    old_sent = ('at the plan found under the doubled flexibility price the final poll evaluated twelve neighbours, which '
                'do not positively span the space, and the plan is reported as better than each of them, seven '
                'determinately and five within resolution, not as a mesh-local optimum.')
    par_norm = _apply(par_norm, old_sent, R3_INLINE[[x[0] for x in R3_INLINE].index('R3-B.d(1)')][1])
    par_norm = _apply(par_norm, 'for the first search under the doubled flexibility price,',
                      'for the first search under the doubled flexibility price, '
                      + R3_INLINE[[x[0] for x in R3_INLINE].index('R3-B.d(2)')][1])
    for blk, (bid, _m) in zip(b2, want2):
        out.append(block_status(st, bid + ' (round-2 text as written)', blk))
    out.append(block_status(st, 'R2-A.2 Algorithm 1 (with round-3 B.a-c applied)', alg_final))
    out.append(block_status(st, 'R2-A.3 paragraph (with round-3 B.d applied)', par_norm))
    for item_id, new, old in R2_INLINE:
        out.append(block_status(st, item_id, new, old))
    # round 3
    for blk, (bid, _m) in zip(b3, want3):
        out.append(block_status(st, bid, blk))
    for item_id, new, old in R3_INLINE:
        out.append(block_status(st, item_id, new, old))
    # C.1: the table equals W167's fragment (from \begin{table} on)
    frag = _read(os.path.join(P53, 'w167_nomenclature_years', 'year_tables', 'tab_investment_cost.tex'))
    frag_table = frag[frag.index('\\begin{table}'):]
    out.append(block_status(st, 'R3-C.1 table = W167 fragment', frag_table))
    adj = []
    for aid, text in ADJACENT:
        adj.append({'id': aid, 'text': text, 'lines': st.find_all(text), 'present': bool(st.find_all(text))})
    return {'items': out, 'adjacent_text_checks': adj,
            'round2_sha256': _sha(ROUND2), 'round3_sha256': _sha(ROUND3)}


# ---------------------------------------------------------------------------------------------------------------------
#  C. CONFIRM comments
# ---------------------------------------------------------------------------------------------------------------------
def check_confirm_comments(tex):
    out = []
    for i, line in enumerate(tex.splitlines(), start=1):
        m = re.match(r'\s*%\s*\[CONFIRM\s*—\s*(W\d+)\]', line)
        if m:
            out.append({'line': i, 'task': m.group(1), 'text': line.strip()})
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  D. code identity
# ---------------------------------------------------------------------------------------------------------------------
def check_code_identity():
    results_files = []
    for root in CAMPAIGN_ROOTS:
        results_files += sorted(glob.glob(os.path.join(root, '**', 'campaign_results.json'), recursive=True))
    heads = {}
    for path in results_files:
        for h in set(_walk(_read_json(path), 'git_head_at_run')):
            heads.setdefault(h, []).append(path)
    pathspec = list(PRODUCTION_FILES) + list(CASE_DATA)
    per_head = {}
    for h in sorted(heads):
        stat = _git(['diff', '--stat', h, 'HEAD', '--'] + pathspec).strip()
        per_head[h] = {'campaign_results': heads[h], 'diff_stat': stat or 'NO CHANGE'}
    control = _git(['diff', '--stat', CONTROL_HEAD, 'HEAD', '--'] + pathspec).strip().splitlines()
    search_heads = {}
    for name, d in SEARCH.items():
        h = _read_json(d['results'])['git_head_at_run']
        stat = _git(['diff', '--stat', h, 'HEAD', '--'] + pathspec).strip().splitlines()
        search_heads[name] = {'git_head_at_run': h, 'last_line': stat[-1] if stat else 'NO CHANGE'}
    out = {'repo_head': _git(['rev-parse', 'HEAD']).strip(), 'n_campaign_results': len(results_files),
           'n_distinct_heads': len(heads),
           'n_heads_with_change': sum(1 for v in per_head.values() if v['diff_stat'] != 'NO CHANGE'),
           'pathspec': pathspec, 'per_head': per_head,
           'control': {'head': CONTROL_HEAD, 'last_line': control[-1] if control else 'NO CHANGE'},
           'search_heads_informational': search_heads}
    if out['n_heads_with_change'] or out['control']['last_line'] == 'NO CHANGE':
        raise SystemExit(f"code identity check failed: changed={out['n_heads_with_change']} control={out['control']}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  E. record checks
# ---------------------------------------------------------------------------------------------------------------------
def check_search_records():
    out = {}
    for name, d in SEARCH.items():
        res = _read_json(d['results'])
        polls = []
        for r in res.get('poll_history', []):
            dirs = [c for c in r['candidates'] if c['poll_part'] == 'direction']
            polls.append({'poll_index': r['poll_index'], 'delta': r['poll_size_delta'], 'n_directions': len(dirs),
                          'n_directions_inadmissible': sum(c['disposition'] == 'rejected_infeasible' for c in dirs),
                          'n_directions_admissible': sum(c['disposition'] != 'rejected_infeasible' for c in dirs),
                          'direction_dispositions': [c['disposition'] for c in dirs],
                          'n_new_evaluations': r['n_new_evaluations'], 'decision': r['decision']})
        out[name] = {'git_head_at_run': res['git_head_at_run'], 'termination_reason': res['termination']['reason'],
                     'n_new_evaluations': res.get('n_new_evaluations'), 'budget_new_evaluations': d['budget'],
                     'polls': polls,
                     'every_direction_inadmissible_at_every_poll': all(p['n_directions_admissible'] == 0 for p in polls),
                     'stopped_by_budget': res['termination']['reason'] == 'evaluation_budget_exhausted'}
    return out


def check_search_tail():
    out = {}
    for name, d in SEARCH.items():
        spec = _read_json(d['spec'])
        head = _read_json(d['results'])['git_head_at_run']
        counts = {}
        for f, pat in (('shared_resources_planning.py', 'def _apply_convergence_depth_tail'),
                       ('admm_parameters.py', 'convergence_depth_tail'),
                       ('p515_s44_campaign_harness.py', 'convergence_depth_tail')):
            src = _git(['show', f'{head}:{f}'])
            counts[f] = src.count(pat)
        out[name] = {'spec_configuration_declares_convergence_depth_tail':
                     'convergence_depth_tail' in (spec.get('configuration') or {}),
                     'spec_text_mentions_tail': 'convergence_depth_tail' in json.dumps(spec),
                     'git_head_at_run': head, 'tail_code_occurrences_at_head': counts}
    return out


def check_certificates():
    ft = _read_json(FROZEN_TABLES)
    cells = {}
    for key in ('cells', 'cells_appended_w160'):
        block = ft['tables'].get(key) or {}
        items = block.items() if isinstance(block, dict) else ((c.get('cell') or c.get('label'), c) for c in block)
        for name, cell in items:
            if isinstance(cell, dict):
                cells[name] = cell
    table = {}
    for name, cell in cells.items():
        cs = cell.get('certifying_spec')
        ver = None
        if isinstance(cs, dict):
            ver = cs.get('criterion_version') if cs.get('criterion_version') is not None else cs.get('version')
            if cs.get('series') == 'frozen_s53_resettle_ext_spec':
                ver = cs.get('criterion_version')
        table[name] = {'status': cell.get('status'), 'k_star': cell.get('k_star'), 'criterion_version': ver,
                       'series': cs.get('series') if isinstance(cs, dict) else None}
    v6 = _read_json(V6_FROM_RECORDS)['item1_v6_on_every_record']
    recs = {}
    for name, r in v6.items():
        cs = r.get('certifying_spec') or {}
        recs[name] = {'committed_criterion': cs.get('criterion') if cs else None,
                      'committed_k_star': cs.get('k_star') if cs else None, 'excluded': cs.get('excluded') if cs else None,
                      'v6_status': (r.get('v6') or {}).get('status'), 'v6_k_star': (r.get('v6') or {}).get('k_star'),
                      'v6_equals_committed_k_star': (cs.get('k_star') == (r.get('v6') or {}).get('k_star')) if cs else None}
    w168 = _read_json(W168_DECISION)
    return {'frozen_table_cells': table, 'w142_v6_from_records': recs,
            'w168_x0_decision_primary': {k: (w168.get('decision_primary') or {}).get(k)
                                         for k in ('status', 'k_star', 'branch', 'version')}}


# ---------------------------------------------------------------------------------------------------------------------
#  F. code anchors
# ---------------------------------------------------------------------------------------------------------------------
ANCHORS = (
    # section 2.1 / Algorithm 1
    ('na_scenario', 'model_construction_helpers.py', 'def sess_na_scenario(m):'),
    ('row_is_duplicate', 'model_construction_helpers.py', 'def sess_row_is_duplicate(m, s_m, s_o):'),
    ('commitment_terms', 'model_construction_helpers.py', 'def add_scenario_commitment_terms(model, network, params, premium_alpha=0.0,'),
    ('single_scenario_guard', 'model_construction_helpers.py', 'if n_scenarios == 1:'),
    ('expected_pf_def', 'model_construction_helpers.py', 'def dn_interface_expected_pf_p_def(m, p, network):'),
    ('tso_pc_fix', 'shared_resources_planning.py', 'fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)'),
    ('tso_delta_bound', 'shared_resources_planning.py',
     'tso_model[year][day].interface_delta_p[dn, s_m, s_o, p].setlb(-interface_transf_rating)'),
    ('n_vars', 'p515_s47_phase_b_record.py', 'N_VARS = 2 * len(ACTIVE_NODES) + 1'),
    ('delta_0', 'p515_s47_phase_b_record.py', 'DELTA_0 = 4'),
    ('max_new_20', 'p515_s47_phase_b_record.py', 'MAX_NEW_EVALUATIONS = 20'),
    ('max_polls_60', 'p515_s47_phase_b_record.py', 'MAX_POLLS = 60'),
    ('barrier_per_poll', 'p515_s47_phase_b_record.py', 'BARRIER_STOP_PER_POLL = 2'),
    ('barrier_overall', 'p515_s47_phase_b_record.py', 'BARRIER_STOP_OVERALL = 3'),
    ('completion_cap', 'p515_s47_phase_b_record.py', 'COMPLETION_CAP = 30'),
    ('project_direction_round', 'p515_s47_phase_b_record.py', 'return tuple(_round_half_away(delta * a / m) for a in h)'),
    ('poll_directions', 'p515_s47_phase_b_record.py', 'def poll_directions(k, delta, n=N_VARS, t0=HALTON_T0, design=POLL_DESIGN):'),
    ('resolution', 'p515_s47_phase_b_record.py', 'return max(bar_x + bar_inc, sigma_q)'),
    ('initial_incumbent', 'p515_s47_phase_b_record.py', 'def initial_incumbent(lattice, cache, x0_eval_key, sigma_q):'),
    ('poll_loop', 'p515_s47_phase_b_record.py', 'for k in range(max_polls):'),
    ('cap_check', 'p515_s47_phase_b_record.py', "if completion is not None and completion['over_cap']:"),
    ('budget_check', 'p515_s47_phase_b_record.py', 'if n_new + len(new) > max_new_evaluations:'),
    ('cache_hit', 'p515_s47_phase_b_record.py', "entry['disposition'] = 'cache_hit'"),
    ('barrier_rule', 'p515_s47_phase_b_record.py',
     'if barrier_this_poll >= BARRIER_STOP_PER_POLL or n_barrier_new >= BARRIER_STOP_OVERALL:'),
    ('success_double', 'p515_s47_phase_b_record.py', 'delta *= 2'),
    ('unit_failure_terminate', 'p515_s47_phase_b_record.py', "termination = {'reason': 'mesh_local_optimum_unit_poll_failed', 'poll_size_reached': delta}"),
    ('halve', 'p515_s47_phase_b_record.py', 'delta = max(DELTA_MIN, delta // 2)'),
    ('s53_max_new_60', 'p515_s53_f2_certificate.py', 'MAX_NEW_EVALUATIONS = 60'),
    ('s53_max_polls', 'p515_s53_f2_certificate.py', 'MAX_POLLS = PB.MAX_POLLS'),
    ('s53_directions_2n', 'p515_s53_f2_certificate.py', 'def directions_2n(k, delta=DELTA_UNIT):'),
    ('s53_completion_trigger', 'p515_s53_f2_certificate.py', 'triggered = n_distinct < MIN_FEASIBLE_POLL_POINTS'),
    ('s53_snap_key', 'p515_s53_f2_certificate.py', 'def snap_key(lattice, z, r, z_inc):'),
    ('s53_certificate_statement', 'p515_s53_f2_certificate.py', 'CERTIFICATE_STATEMENT = ('),
    ('harness_bar', 'p515_s44_campaign_harness.py', 'def _max_step_last_n(rows, n=BAR_WINDOW):'),
    ('harness_status', 'p515_s44_campaign_harness.py',
     "status = 'certified' if certified else ('not_certified' if rows else 'no_trajectory')"),
    ('harness_tail_default_off', 'p515_s44_campaign_harness.py',
     "in the record. The tail stays OFF by default (`admm_parameters.ADMMParameters.convergence_depth_tail`); a launcher"),
    # section 2.2.7
    ('p_hat', 'settling_criterion_v2.py', 'p_hat = T[-1][0] - T[-3][0]'),
    ('eps0', 'settling_criterion.py', 'EPS0 = TAU / 100.0'),
    ('k_excl', 'settling_criterion.py', 'K_EXCL = 3'),
    ('k_excl_step3', 'settling_criterion_v2.py', 'if k < self.k0 + K_EXCL:'),
    ('lapse_reset', 'settling_criterion_v2.py', 'self.k0 = None'),
    ('osc_window_inside_run', 'settling_criterion_v2.py',
     "a_parts.update({'P_hat': p_hat, 'W': w, 'window': [lo, k], 'window_inside_run': lo >= self.k0})"),
    ('mono_window_start', 'settling_criterion_v2.py', "'lo_ge_k0_plus_K_EXCL': lo >= self.k0 + K_EXCL,"),
    ('mono_steps_decreasing', 'settling_criterion_v2.py', "'steps_decreasing': bool(max(steps) < EPS0 or m2 < m1)})"),
    ('gap_at_certifying_cycle', 'settling_criterion_v2.py', 'gap_ok = t_sum is not None and abs(t_sum) <= GAP_BOUND'),
    ('swing_floor', 'settling_criterion_v6.py', 'SWING_FLOOR = TAU / 10.0'),
    ('v6_mono_veto', 'settling_criterion_v6.py', "vetoed.append({'branch': 'monotone', 'window': list(b_parts['window']), 'W': self.l_mono,"),
    ('v6_threshold_ge', 'settling_criterion_v6.py', "return abs(margin) >= thr, thr, ('3 x larger bar' if three >= two_tau else '2 TAU')"),
    ('v5_esso_tolerances', 'settling_criterion_v5.py', "'esso': {'tol': 1e-10, 'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4},"),
    ('slack_signed', 'p515_s53_w132_resettle_v3_campaign.py', "rep.update({'Q_N_old': q_n_old, 's_signed': (q_end - q_n_old) if q_end is not None else None,"),
    ('uncertified_bar', 'p515_s53_w132_resettle_v3_campaign.py', 'det = abs(d_q) > bar and m_cc > bar'),
    ('w118_hold', 'p515_s53_w118_resettle_hooks.py', 'return self.first_pass is not None and c > self.first_pass'),
    ('w101_hold', 'p515_s53_w101_settling_continuation_hooks.py', 'hold = iter > st.n'),
    ('objective_rule', 'model_construction_helpers.py', 'def objective_function_rule(model, params):'),
    ('generation_cost', 'model_construction_helpers.py', 'def generation_cost(model, network, s_m, s_o, params):'),
    ('res_curtailment_zero_dso', 'shared_resources_planning.py', 'dso_model[year][day].penalty_gen_curtailment.set_value(0.00)'),
    ('cohort_window', 'shared_energy_storage_data.py',
     'tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y_inv]]))'),
    ('cohort_num_years', 'shared_energy_storage_data.py', 'num_years = shared_ess_data.years[repr_years[y_inv]]'),
    # section 2.3
    ('network_pnet', 'model_construction_helpers.py',
     'return m.shared_es_pnet[e, s_m0, s_o0, p] == m.shared_es_pch[e, s_m0, s_o0, p] - m.shared_es_pdch[e, s_m0, s_o0, p]'),
    ('esso_pnet', 'shared_energy_storage_data.py',
     'model.energy_storage_operation_agg.add(model.es_pnet[y, d, p] == agg_pnet + model.slack_es_pnet_up[y, d, p] - model.slack_es_pnet_down[y, d, p])'),
    ('esso_circle', 'shared_energy_storage_data.py',
     'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2)'),
    ('closure_slack_ub', 'model_construction_helpers.py',
     'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE'),
    ('soc_pre_day', 'model_construction_helpers.py', 'soc_prev = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC'),
    ('closure_penalty_in_obj', 'model_construction_helpers.py', 'obj += model.total_ess_complementarity_penalties'),
    ('esso_build_subproblem', 'shared_energy_storage_data.py', 'def build_subproblem(self):'),
    ('soh_point_default', 'shared_energy_storage_data.py', "AVAILABLE_ENERGY_SOH_POINT_DEFAULT = 'end'"),
    # Appendix A
    ('settlement_weight_param', 'model_construction_helpers.py',
     'model.interface_settlement_weight = pe.Param(initialize=0.00, mutable=True)'),
    ('settlement_in_obj', 'model_construction_helpers.py', 'obj += model.interface_settlement_weight * model.interface_settlement'),
    ('settlement_weight_tso', 'shared_resources_planning.py', 'model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('settlement_weight_dso', 'shared_resources_planning.py', 'dso_model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('row18_activate_call', 'shared_resources_planning.py', '_activate_row18_with_settlement(dso_model[year][day])'),
    ('prepare_dso_call', 'shared_resources_planning.py', '_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)'),
    ('convert_dso', 'shared_resources_planning.py',
     'update_distribution_models_to_admm(planning_problem, dso_models, admm_parameters, objective_scale)'),
    ('z_init_call', 'shared_resources_planning.py', '_initialize_shared_ess_consensus(planning_problem, consensus_vars)'),
    ('interface_dual_init', 'shared_resources_planning.py', 'planning_problem.update_interface_power_flow_variables('),
    ('esso_al_kappa', 'shared_resources_planning.py',
     'obj += models[node_id].admm_esso_al_scale * (models[node_id].dual_p_req[y, d, p] * constraint_p_req)'),
    ('z_update', 'shared_resources_planning.py', 'z_new = numerator / denominator'),
    ('interface_dual_gate', 'shared_resources_planning.py', 'if update_tn and tso_succeeded and dso_succeeded:'),
    ('boyd_s_ess_per_agent', 'shared_resources_planning.py', 's_agent_rho = rho_ess[agent] * a[agent] * dz_ess'),
    ('boyd_y_pf_dso_only', 'shared_resources_planning.py', 'y_pf = lambda_dso_pf / s_base_dso'),
    ('aa_step_call', 'shared_resources_planning.py', 'aa_record = _anderson_acceleration_cycle_step('),
    ('penalty_update_call', 'shared_resources_planning.py',
     'penalty_actions, penalties_before, penalties_after, gamma_before, gamma_after, rho_freeze_active, freeze_state = _update_admm_penalties('),
    ('aa_clear_rho', 'shared_resources_planning.py', 'aa_rho_change_record = aa_state.clear_for_rho_change(iter, aa_rho_changed)'),
    ('tail_next_call', 'shared_resources_planning.py', 'convergence_depth_tail_active_next = _convergence_depth_tail_next_state('),
    ('production_exit', 'shared_resources_planning.py',
     'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)'),
    ('exit_break', 'shared_resources_planning.py', 'ADMM converged in {iter} iteration(s).'),
    ('loop', 'shared_resources_planning.py', 'for iter in range(1, admm_parameters.num_max_iters + 1):'),
    ('aa_off_branch', 'admm_anderson_acceleration.py', "action='off (all channels within Boyd tolerance)',"),
    ('aa_clear_rho_def', 'admm_anderson_acceleration.py', 'def clear_for_rho_change(self, cycle, channels_changed):'),
    ('aa_skip_failure_def', 'admm_anderson_acceleration.py', 'def skip_on_failure(self, cycle):'),
    ('aa_interface_z_is_tso_copy', 'admm_anderson_acceleration.py',
     "raw = consensus_vars['pf']['tso']['current'][node_id][year][day][pt][p]"),
    ('w118_aa_forced_off', 'p515_s53_w118_resettle_hooks.py', "forced['all_boyd_pass'] = True"),
    ('tail_default_off', 'admm_parameters.py', 'self.convergence_depth_tail = {'),
    ('network_recoverable', 'network.py', 'def _is_recoverable_network_failure(result, params):'),
    ('network_tier2', 'network.py', "tier2_options['mu_strategy'] = 'adaptive'"),
    ('esso_recoverable', 'shared_energy_storage_data.py', 'def _is_recoverable_shared_ess_failure(result, params, node_id):'),
    ('esso_tol_overrides', 'shared_energy_storage_data.py', "ESSO_TOL_OVERRIDES = {'tol': 1e-10, 'acceptable_tol': 1e-9}"),
    ('plan_into_network_init', 'shared_resources_planning.py', 'update_data_with_candidate_solution(candidate_solution)'),
    ('capacities_publish', 'shared_resources_planning.py',
     'sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)'),
)


def check_anchors():
    out = {}
    for anchor_id, path, text in ANCHORS:
        src = _read(path)
        hits = [i for i, line in enumerate(src.splitlines(), start=1) if text in line]
        if not hits:
            raise SystemExit(f'anchor {anchor_id} not found in {path}: {text!r}')
        out[anchor_id] = {'file': path, 'text': text, 'lines': hits}
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'refusing to overwrite {OUT_JSON} / {OUT_MANIFEST}')
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manuscript = check_manuscript()
    tex = _read(MAIN_TEX)
    st = Stripped(tex)
    result = {
        'schema': 'p515_s53_w174b_reaudit_checks_v1',
        'stage': 'P5.15 Addendum 70 W174b -- re-audit of the W171b and W172 rows at Overleaf 260bd83; correction-block '
                 'presence; CONFIRM comments; code identity',
        'started_utc': started,
        'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'interpreter': sys.executable,
        'A_manuscript': manuscript,
        'B_correction_blocks': check_blocks(st),
        'C_confirm_comments': check_confirm_comments(tex),
        'D_code_identity': check_code_identity(),
        'E1_search_records': check_search_records(),
        'E2_search_tail': check_search_tail(),
        'E3_certificates': check_certificates(),
        'F_code_anchors': check_anchors(),
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
    blocks = result['B_correction_blocks']['items']
    counts = {}
    for b in blocks:
        counts[b['status']] = counts.get(b['status'], 0) + 1
    print(f"W174b checks: manuscript {manuscript['clone_head']} sha256 {manuscript['main_tex_sha256'][:8]} "
          f"lines {manuscript['main_tex_lines']}")
    print(f"blocks: {counts}")
    for b in blocks:
        if b['status'] not in ('present_verbatim', 'old_text_absent') or b.get('old_text_absent') is False:
            print(f"  {b['id']}: {b['status']} old_absent={b.get('old_text_absent')} lines={b.get('lines') or b.get('region_lines')}")
            for d in b.get('word_diff', [])[:12]:
                print(f"     {d['op']}: expected [{d['expected'][:160]}] found [{d['found'][:160]}]")
    for a in result['B_correction_blocks']['adjacent_text_checks']:
        print(f"  adjacent {a['id']}: present={a['present']} lines={a['lines']}")
    ci = result['D_code_identity']
    print(f"code identity: {ci['n_campaign_results']} campaign results, {ci['n_distinct_heads']} distinct heads, "
          f"{ci['n_heads_with_change']} changed; control {ci['control']}; search heads {ci['search_heads_informational']}")
    for n, v in result['E1_search_records'].items():
        print(f"search {n}: termination {v['termination_reason']}; new {v['n_new_evaluations']}/{v['budget_new_evaluations']}; "
              f"every direction inadmissible at every poll {v['every_direction_inadmissible_at_every_poll']}; "
              f"admissible per poll {[p['n_directions_admissible'] for p in v['polls']]}")
    print(f"search tail: {json.dumps(result['E2_search_tail'])}")
    print(f"guards: {result['guards']}")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
