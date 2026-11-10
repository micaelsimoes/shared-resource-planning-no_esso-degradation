#!/usr/bin/env python3
"""Generate the machine-readable supplement (Addendum 89 ruling 8; Planner task W188 (b)). ZERO SOLVES.

WHAT IT WRITES. One deterministic zip, its hash inventory and a generation record, into a NEW directory:
    manuscript_review/supplement/                       (version 1; created if absent)
    manuscript_review/supplement_v<N>/                  (--force-new-version: the next free version N >= 2; it records
                                                         its predecessor's zip sha256)
The directory must be absent or hold nothing but this generator's launch log ``gen_launch.log``; anything else is
refused (an existing supplement is never overwritten). ``--out-dir DIR`` writes a scratch copy OUTSIDE the repository
(used for the determinism check); inside the repository only the canonical / versioned directories are accepted.

ZIP CONTENTS (entries sorted; every entry dated 1980-01-01 00:00:00, mode 0644, ZIP_STORED -- no compressor in the
byte path, so the bytes do not depend on the zlib build):
  appendix_E/E20_comparisons.csv ... E24_gap_refused.csv   the full Tables E.20-E.24 of Appendix E
  appendix_E/E_table_notes.csv                             the numbers printed in the notes of E.20, E.22 and E.23
  appendix_E/E20_E21_groups.csv                            the group titles of E.20 / E.21 (LaTeX source, verbatim)
  appendices_B_D/*.csv                                     the Appendix B-D profiles and network tables, from the case
                                                           files under data/SRP1/
  README_PLACEHOLDER.md                                    column definitions generated from COLUMN_SPECS below, and
                                                           marked placeholders for the expert's prose (tau, the
                                                           objective convention, the certification rule's parameters,
                                                           each filled from the code / constants where they exist)
  SHA256SUMS.txt                                           sha256 of every other entry
Outside the zip: ``supplement_v<N>_record.json`` (inventory of every entry and of the zip, input hashes, provenance,
guard counts), written through the shared writer gate_result_io.

HOW THE NUMBERS ARE OBTAINED (nothing is typed).
  * Appendix E: manuscript_review/gen_appE.py is EXECUTED (imported as a module, as verify_appE.py does) into a
    temporary file; its own row builders (e1_row / e2_row) and its generated LaTeX give every table cell. Each cell is
    the printed cell with the LaTeX markup removed: '---' -> empty, '$-$' -> '-', thousands separator '\\,' removed in
    numeric columns, the marks ($^{b}$, $^{f}$, $^{t}$, $^{s}$, $^{u}$, $^{n}$, $^{\\Delta}$, E.23's $^{b}$) moved to
    explicit yes/no columns. gen_appE.py reads export_W160/*.csv, which W160 generated from the frozen tables
    frozen_step6_tables_v1_590088fe.json.
  * Appendices B-D: the case files, read with production's own readers where production has one
    (shared_resources_planning._read_market_base_profiles, network_data._read_network_base_profiles) and with json for
    the network JSON files. Values are written at full precision (repr of the stored float; integers as integers).

GUARDS. SolveProfileGuard(permitted=()) is installed BEFORE any project import; pickle.load / pickle.loads are blocked
for the whole run; the guard is verified at exactly 0 solves / 0 process launches at the end. The run refuses to start
beside another copy of itself. Run it attached with both streams captured:
    mkdir manuscript_review/supplement && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/gen_supplement.py \\
        > manuscript_review/supplement/gen_launch.log 2>&1
Exit: 0 = written and every check holds; 1 = precondition or guard fault (nothing written).
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

GUARD = SolveProfileGuard(permitted=(), label='W188b supplement generator (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('gen_supplement: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('gen_supplement: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import numpy as np  # noqa: E402
import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402  (production reader: _read_market_base_profiles)
import network_data as ND  # noqa: E402  (production reader: _read_network_base_profiles)

pickle.load, pickle.loads = _blocked_load, _blocked_loads   # re-bound after the imports

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_REL = os.path.join('manuscript_review', 'gen_supplement.py')
MR = 'manuscript_review'
GEN_REL = os.path.join(MR, 'gen_appE.py')
APPX_REL = os.path.join(MR, 'appendixE_evaluation_record.tex')
EXPORT_DIR = os.path.join(MR, 'export_W160')
EXPORT_FILES = ('MANIFEST.md', 'README.md', 'README_v2_note.md', 'T1.csv', 'T1.tex', 'T10.csv', 'T10.tex', 'T2.csv',
                'T2.tex', 'T5.csv', 'T5.tex', 'T6.csv', 'T6.tex', 'T9.csv', 'T9.tex', 'paragraphs_v2.md')
FZ_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
SPEC_V6_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6',
                           'frozen_s53_resettle_spec_v6_96c23404.json')
CANONICAL_DIR = os.path.join(MR, 'supplement')
LAUNCH_LOG = 'gen_launch.log'
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = 0o100644 << 16
NAME_STEM = 'SRP1_supplement'

DATA = os.path.join('data', 'SRP1')
MARKET_XLSX = os.path.join(DATA, 'MarketData', 'SRP1_market_data.xlsx')
TN = 'case9'
ADNS = (('case33_1', 5), ('case33_2', 7), ('case33_3', 9))      # (network, TN connection node): data/SRP1/SRP1.json
CONFIG_FILES = (os.path.join(DATA, 'SRP1.json'), os.path.join(DATA, 'SRP1_5x3y_1x1.json'))
PRINTED_YEARS = (2025, 2028, 2031, 2034, 2037)   # the capacity columns printed in Appendices C-D (multi-scenario years)
ALL_CASE_YEARS = (2025, 2028, 2030, 2031, 2034, 2035, 2037)   # SRP1.json and SRP1_5x3y_1x1.json years together
HOURS = [f'h{h:02d}' for h in range(24)]
GEN_TYPE_LABEL = {'WIND': 'Wind', 'PV': 'Solar PV'}       # the labels printed in the RES-generator tables
BUS_TYPE_LABEL = {3: 'Ref.', 1: 'PQ'}                       # the labels printed in tab:cs1_ieee33_buses


def net_json(net, year):
    return os.path.join(DATA, net, f'{net}_{year}.json')


def net_xlsx(net):
    return os.path.join(DATA, net, f'{net}_operational_data.xlsx')


# ======================================================================================================================
#  column specifications -- the README's column definitions are generated from these, and the verifier checks every
#  CSV header against them. (name, unit, definition)
# ======================================================================================================================
YN = 'yes / no'
COLUMN_SPECS = {
    'appendix_E/E20_comparisons.csv': {
        'table': 'Table E.20 (tab:appE_claims): every pairwise comparison of the record',
        'source': 'gen_appE.py e1_row over gen_appE.E1, from export_W160/T1.csv and T10.csv',
        'columns': [
            ('no', '', 'row number of Table E.20'),
            ('group', '', 'row group (a)-(d) of Table E.20; titles in E20_E21_groups.csv'),
            ('comparison', '', 'the comparison as printed (plan node: MVA/MWh; "->" from the smaller to the larger plan)'),
            ('claim_id', '', 'claim identifier in the frozen tables (export_W160 T1.csv / T10.csv column "claim")'),
            ('claim_source', '', 'export table the row is read from: T1 (gross and net) or T10 (gross only)'),
            ('eval_ref', '', 'reference evaluation: row number of Table E.21 (E21_evaluations.csv "no")'),
            ('eval_ref_cell', '', 'cell identifier of the reference evaluation (E21_evaluations.csv "cell_id")'),
            ('eval_ref_uncertified', YN, 'printed mark u: the reference evaluation is uncertified'),
            ('eval_ref_flagged', YN, 'printed mark f: the reference evaluation is a flagged certificate (turning point at a non-clean cycle)'),
            ('eval_other', '', 'other evaluation: row number of Table E.21'),
            ('eval_other_cell', '', 'cell identifier of the other evaluation'),
            ('eval_other_uncertified', YN, 'printed mark u on the other evaluation'),
            ('eval_other_flagged', YN, 'printed mark f on the other evaluation'),
            ('delta_gross_kEUR', 'k EUR', 'Delta on Q, the gross operational cost, contracted interface settlement excluded (sign convention per group)'),
            ('threshold_or_bar_kEUR', 'k EUR', 'determinacy threshold max(3b, 2 tau) between certified evaluations, or the uncertified bar 3 max(|gap|, |slack|)'),
            ('multiple_gross', '', '|Delta gross| / threshold or bar'),
            ('verdict_gross', '', 'det. = determinate; w.r. = within resolution; w.b. = within the uncertified bar'),
            ('delta_net_kEUR', 'k EUR', 'Delta net of the terminal salvage credit; empty: not evaluated net'),
            ('multiple_net', '', '|Delta net| / the same threshold or bar; empty: not evaluated net'),
            ('verdict_net', '', 'verdict of the net difference (abbreviations as verdict_gross); empty: not evaluated net'),
            ('over_budget', YN, 'printed mark b: the comparison involves an investment above the 1 M EUR budget (a valuation probe outside the admissible set)'),
            ('over_budget_plans', '', 'the plan(s) carrying the mark b, "node: MVA/MWh", separated by "; "'),
            ('increment_row', YN, 'printed mark Delta: the increase of value - I from the smaller to the larger plan'),
            ('not_unit_neighbour', YN, 'printed mark n: not a unit neighbour of the returned plan'),
            ('bitwise_twin_row', YN, 'printed mark t: the row uses the bitwise twin of evaluation 2'),
        ]},
    'appendix_E/E21_evaluations.csv': {
        'table': 'Table E.21 (tab:appE_evaluations): every recourse evaluation the comparisons use',
        'source': 'gen_appE.py e2_row over gen_appE.G, from export_W160/T2.csv',
        'columns': [
            ('no', '', 'evaluation number (row of Table E.21)'),
            ('group', '', 'row group (a)-(f) of Table E.21; titles in E20_E21_groups.csv'),
            ('cell_id', '', 'cell identifier in the frozen tables (export_W160 T2.csv column "cell")'),
            ('eval_key', '', 'evaluation key prefix (instance identifier; full key in the frozen JSON)'),
            ('candidate_key', '', 'candidate (investment vector) key prefix'),
            ('plan', '', 'investment plan, node: converter rating MVA / energy capacity MWh; "x = 0": no storage'),
            ('year', '', 'installation year; empty for the plan without storage'),
            ('condition', '', 'flexibility-price multiplier m (base: m = 1) or ageing calibration'),
            ('status', '', 'cert. / cert., osc. / cert., mon. (certified on the oscillatory or monotone branch), the cause that left the evaluation uncertified (gap clause, lapse reset, growth test), or superseded'),
            ('flagged', YN, 'printed mark f: flagged certificate (a turning point at a non-clean cycle)'),
            ('bitwise_twin', YN, 'printed mark t: bitwise twin of evaluation 2'),
            ('superseded', YN, 'printed mark s: superseded by evaluation 15'),
            ('over_budget', YN, 'printed mark b: investment above the 1 M EUR budget (valuation probe outside the admissible set)'),
            ('k0', 'cycle', 'first residual pass; empty: not tabulated'),
            ('kstar_or_end', 'cycle', 'certification cycle k*, or the cycle at which an uncertified evaluation stopped'),
            ('range_over_tau', '', 'range of Q over the certifying window / tau (certified evaluations)'),
            ('band_kEUR', 'k EUR', 'band b that enters the determinacy threshold (certified evaluations)'),
            ('gap_kEUR', 'k EUR', 'magnitude of the priced interface-consensus gap (uncertified evaluations)'),
            ('slack_kEUR', 'k EUR', 'magnitude of the settling slack (uncertified evaluations)'),
            ('non_clean_cycles_after_k0', 'cycles', 'cycles after k0 with a local solve that was not clean'),
            ('replay_bitwise_through', 'cycle', 'last cycle a repeated evaluation replayed its earlier record bitwise'),
            ('rule', '', 'run: the certification rule was applied in the run; record: applied to the committed record of an earlier run; empty: superseded'),
        ]},
    'appendix_E/E22_neighbours.csv': {
        'table': 'Table E.22 (tab:appE_neighbours): neighbours of the plan without storage re-evaluated under the certification rule',
        'source': 'gen_appE.py generated table, from export_W160/T5.csv (single nodes) and T1.csv (pairs)',
        'columns': [
            ('nodes', '', 'node(s) carrying the smallest unit, 0.25 MVA / 0.5 MWh'),
            ('year', '', 'installation year'),
            ('eval', '', 'evaluation number (Table E.21)'),
            ('cell_id', '', 'cell identifier of the evaluation'),
            ('delta_kEUR', 'k EUR', 'Delta = F(x) - F(0), gross operational cost, contracted interface settlement excluded'),
            ('threshold_kEUR', 'k EUR', 'determinacy threshold'),
            ('multiple', '', '|Delta| / threshold (every difference is determinate)'),
        ]},
    'appendix_E/E23_benchmark.csv': {
        'table': 'Table E.23 (tab:appE_benchmark): static no-reverse-flow arrangements at the plan without storage, by solver start',
        'source': 'gen_appE.py generated table, from export_W160/T6.csv',
        'columns': [
            ('arrangement', '', 'Coordinated, Passive NRF or Price-taker NRF'),
            ('start', '', 'solver start: cold (default), perturbed, warm (from the certified coordinated evaluation); empty for the coordinated row'),
            ('reported_start', YN, 'printed mark b of Table E.23: the start reported in the main-text benchmark table (not the budget mark)'),
            ('Q_kEUR', 'k EUR', 'Q, gross operational cost, contracted interface settlement excluded, after the consistency pass'),
            ('nrf_exceedances_n', 'network-hours', 'network-hours (of 864) whose interface flow exceeds the no-reverse-flow limit'),
            ('nrf_largest_excess_pu', 'p.u.', 'largest exceedance'),
            ('consistency_pass_kEUR', 'k EUR', 'change of Q from the first phase to the consistency pass'),
            ('curtailment_TN_GWh', 'GWh', 'TN renewable curtailment, day-weighted over the horizon, first phase'),
            ('curtailment_DN_GWh', 'GWh', 'DN renewable curtailment, day-weighted over the horizon, first phase'),
        ]},
    'appendix_E/E24_gap_refused.csv': {
        'table': 'Table E.24 (tab:appE_gap): evaluations refused by the gap clause, and two others for comparison',
        'source': 'gen_appE.py generated table, from export_W160/T9.csv',
        'columns': [
            ('eval', '', 'evaluation number (Table E.21)'),
            ('cell_id', '', 'cell identifier of the evaluation'),
            ('plan', '', 'investment plan, as in Table E.21'),
            ('year', '', 'installation year'),
            ('condition', '', 'flexibility-price multiplier or "base"'),
            ('cause', '', 'cause that left the evaluation uncertified'),
            ('over_budget', YN, 'investment above the 1 M EUR budget: the over_budget flag of the same evaluation in E21_evaluations.csv (Table E.24 itself prints no mark)'),
            ('gap_kEUR', 'k EUR', 'priced interface-consensus gap t_sum at the cap (signed)'),
            ('share_n5_pct', '%', 'split of the gap at interface node 5; empty: not recorded'),
            ('share_n7_pct', '%', 'split of the gap at interface node 7; empty: not recorded'),
            ('share_n9_pct', '%', 'split of the gap at interface node 9; empty: not recorded'),
            ('pf_primal_residual_ratio', '', 'power-flow primal residual over its tolerance, last cycle; empty: not recorded'),
            ('lapse_resets', 'cycle', 'cycles of the lapse resets; empty: none'),
        ]},
    'appendix_E/E_table_notes.csv': {
        'table': 'Numbers printed in the notes of Tables E.20, E.22 and E.23',
        'source': 'gen_appE.py note variables (r3, r26, sup, coord, bests), from export_W160/T1.csv, T5.csv and T6.csv',
        'columns': [
            ('table', '', 'E.20, E.22 or E.23'),
            ('item', '', 'item identifier'),
            ('value', '', 'the number (or list) as printed in the note'),
            ('unit', '', 'unit of the value'),
            ('description', '', 'what the value is'),
        ]},
    'appendix_E/E20_E21_groups.csv': {
        'table': 'Group titles of Tables E.20 and E.21',
        'source': 'gen_appE.E1 and gen_appE.G (LaTeX source, verbatim)',
        'columns': [
            ('table', '', 'E.20 or E.21'),
            ('group', '', 'group letter'),
            ('title_latex', '', 'the group title as LaTeX source, verbatim from gen_appE.py'),
        ]},
    'appendices_B_D/B_market_prices.csv': {
        'table': 'Appendix B (fig:cs3_market_prices): wholesale energy and flexibility market price profiles, base year 2025',
        'source': f'{MARKET_XLSX}, sheets Energy and Flexibility (production reader shared_resources_planning._read_market_base_profiles)',
        'columns': [('price', '', 'energy (sheet Energy) or flexibility (sheet Flexibility)'),
                    ('season', '', 'representative day'),
                    ('scenario', '', 'base-profile scenario of the case file')]
                   + [(h, 'as stored', f'price at hour {int(h[1:])}, as stored in the case file') for h in HOURS]},
    'appendices_B_D/C_tn_res_generators.csv': {
        'table': 'Appendix C (tab:cs3_tn_res_generators): TN RES generator locations and installed capacities',
        'source': f'{DATA}/case9/case9_<year>.json, generators of type WIND / PV, field Pmax, years {list(PRINTED_YEARS)}',
        'columns': [('generator_id', '', 'gen_id'), ('node_id', '', 'bus'), ('type', '', 'Wind or Solar PV (type WIND / PV)')]
                   + [(f'capacity_{y}_MW', 'MW', f'installed capacity in {y} (Pmax of case9_{y}.json)') for y in PRINTED_YEARS]},
    'appendices_B_D/C_tn_res_profiles.csv': {
        'table': 'Appendix C (fig:cs3_tn_res_scenarios_2025): TN RES generation base profiles',
        'source': f'{net_xlsx(TN)}, sheet Pg (production reader network_data._read_network_base_profiles)',
        'columns': [('gen_type', '', 'PV or Wind'), ('season', '', 'representative day'),
                    ('scenario', '', 'base-profile scenario of the case file')]
                   + [(h, 'p.u. of installed capacity', f'available generation at hour {int(h[1:])}, as stored') for h in HOURS]},
    'appendices_B_D/D_ieee33_buses.csv': {
        'table': 'Appendix D (tab:cs1_ieee33_buses): IEEE 33-bus test system, bus data (identical in the three ADNs and in every case year)',
        'source': f'{net_json("case33_1", 2025)}, nodes',
        'columns': [('bus_id', '', 'bus_i'), ('bus_type_code', '', 'type (3: reference, 1: PQ)'),
                    ('bus_type', '', 'Ref. or PQ, as printed'), ('g_pu', 'p.u.', 'Gs'), ('b_pu', 'p.u.', 'Bs'),
                    ('v_base_kV', 'kV', 'baseKV'), ('v_max_pu', 'p.u.', 'Vmax'), ('v_min_pu', 'p.u.', 'Vmin')]},
    'appendices_B_D/D_ieee33_branches.csv': {
        'table': 'Appendix D (tab:cs1_ieee33_branches): IEEE 33-bus test system, branch data (branch 1 is the interface transformer)',
        'source': f'{net_json("case33_1", 2025)} transformers + lines (r, x, b, rating); rating also from case33_2 and case33_3',
        'columns': [('branch_id', '', 'branch_id'), ('element', '', 'transformer or line'),
                    ('from_bus', '', 'fbus'), ('to_bus', '', 'tbus'), ('r_pu', 'p.u.', 'r'), ('x_pu', 'p.u.', 'x'),
                    ('b_sh_pu', 'p.u.', 'b'),
                    ('rating_adn_node5_MVA', 'MVA', 'rating in case33_1 (ADN at TN node 5; the printed column)'),
                    ('rating_adn_node7_MVA', 'MVA', 'rating in case33_2 (ADN at TN node 7)'),
                    ('rating_adn_node9_MVA', 'MVA', 'rating in case33_3 (ADN at TN node 9)')]},
    'appendices_B_D/D_ieee33_tn_connection.csv': {
        'table': 'Appendix D (tab:cs1_ieee33_generators): TN connection data (the reference generator of each ADN)',
        'source': 'data/SRP1/case33_<k>/case33_<k>_2025.json, generator of type REF',
        'columns': [('adn_node', '', 'TN node the ADN connects to'), ('network', '', 'case file'),
                    ('generator_id', '', 'gen_id'), ('node_id', '', 'bus'), ('p_max_MW', 'MW', 'Pmax'),
                    ('p_min_MW', 'MW', 'Pmin'), ('q_max_MVAr', 'MVAr', 'Qmax'), ('q_min_MVAr', 'MVAr', 'Qmin'),
                    ('v_set_pu', 'p.u.', 'Vg')]},
    'appendices_B_D/D_ieee33_loads.csv': {
        'table': 'Appendix D (tab:cs1_ieee33_loads): IEEE 33-bus test system, load data',
        'source': (f'{net_json("case33_1", 2025)} loads (bus); data/SRP1/case33_<k>/case33_<k>_operational_data.xlsx, sheet '
                   'Characterization (FL Ratio, identical in the three ADNs; Pc, Qc per ADN)'),
        'columns': [('load_id', '', 'load_id'), ('bus_id', '', 'bus'),
                    ('fl_ratio', 'fraction', 'FL Ratio (printed in %)')]
                   + [c for _n, k in ADNS for c in (
                       (f'pc_base_adn_node{k}_MW', 'MW', f'Pc of the ADN at TN node {k} (base active load; not printed)'),
                       (f'qc_base_adn_node{k}_MVAr', 'MVAr', f'Qc of the ADN at TN node {k} (base reactive load; not printed)'))]},
    'appendices_B_D/D_adn_res_generators.csv': {
        'table': 'Appendix D (tab:cs3_adn_node_5/7/9_res_generators): ADN RES generator locations and installed capacities',
        'source': f'data/SRP1/case33_<k>/case33_<k>_<year>.json, generators of type WIND / PV, field Pmax, years {list(PRINTED_YEARS)}',
        'columns': [('adn_node', '', 'TN node the ADN connects to'), ('network', '', 'case file'),
                    ('generator_id', '', 'gen_id'), ('node_id', '', 'bus'), ('type', '', 'Wind or Solar PV')]
                   + [(f'capacity_{y}_MW', 'MW', f'installed capacity in {y}') for y in PRINTED_YEARS]},
    'appendices_B_D/D_adn_res_profiles.csv': {
        'table': 'Appendix D (fig:cs3_adn_node_5/7/9_res_scenarios_2025): ADN RES generation base profiles',
        'source': 'data/SRP1/case33_<k>/case33_<k>_operational_data.xlsx, sheet Pg',
        'columns': [('adn_node', '', 'TN node the ADN connects to'), ('network', '', 'case file'),
                    ('gen_type', '', 'PV or Wind'), ('season', '', 'representative day'),
                    ('scenario', '', 'base-profile scenario of the case file')]
                   + [(h, 'p.u. of installed capacity', f'available generation at hour {int(h[1:])}, as stored') for h in HOURS]},
    'appendices_B_D/D_adn_load_profiles.csv': {
        'table': 'Appendix D (fig:cs3_adn_node_5/7/9_load_scenarios_2025): ADN load base profiles',
        'source': 'data/SRP1/case33_<k>/case33_<k>_operational_data.xlsx, sheets Pc and Qc',
        'columns': [('adn_node', '', 'TN node the ADN connects to'), ('network', '', 'case file'),
                    ('quantity', '', 'Pc (active) or Qc (reactive), the sheet'), ('load_id', '', 'LoadID'),
                    ('season', '', 'representative day'), ('scenario', '', 'base-profile scenario of the case file')]
                   + [(h, 'p.u. of the base load', f'multiplier of the load base value at hour {int(h[1:])}, as stored') for h in HOURS]},
    'appendices_B_D/D_adn_flexibility_profiles.csv': {
        'table': 'Appendix D (fig:cs3_adn_node_5/7/9_flexibility): ADN flexible-load profiles',
        'source': 'data/SRP1/case33_<k>/case33_<k>_operational_data.xlsx, sheet Flex',
        'columns': [('adn_node', '', 'TN node the ADN connects to'), ('network', '', 'case file'),
                    ('type', '', 'Flexible Load / Maximum Flexible Load / Minimum Flexible Load, [MW]'),
                    ('season', '', 'representative day'), ('scenario', '', 'base-profile scenario of the case file')]
                   + [(h, 'MW', f'value at hour {int(h[1:])}, as stored') for h in HOURS]},
}


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _log(msg):
    print(f'[gen_supplement] {msg}', flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as h:
        return _sha_bytes(h.read())


def _git(*args, cwd=REPO):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=cwd).stdout.strip()


def _committed_clean(rel):
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], capture_output=True, text=True,
                             cwd=REPO).returncode == 0
    return tracked and _git('status', '--porcelain', '--', rel) == ''


def other_copies(script_name):
    """Other python processes running this script (W180a form: the interpreter is argv[0], the script in argv[1:3])."""
    out = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout
    me = os.getpid()
    hits = []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(script_name) for a in argv[1:3]):
            hits.append(ln.strip())
    return hits


def fmt(v):
    """A case-file value at full precision: integers as integers, floats as repr (shortest round-trip)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f'boolean value {v!r} in a case-file column')
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f'non-finite case-file value {v!r}')
        return repr(f)
    if isinstance(v, str):
        return v
    raise TypeError(f'unexpected case-file value {v!r} ({type(v).__name__})')


MARK_RX = re.compile(r'\$\^\{(\\Delta|[a-z])\}\$')


def marks_of(s):
    return [m.replace('\\Delta', 'Delta') for m in MARK_RX.findall(s)]


def unmark(s):
    return MARK_RX.sub('', s).strip()


TEX_LEFT = re.compile(r'[\\$^{}~&]')


def plain_text(s):
    """A printed text cell without its LaTeX markup (marks removed beforehand)."""
    t = s.strip()
    if t == '---':
        return ''
    t = t.replace('$\\boldsymbol{x}=\\boldsymbol{0}$', 'x = 0')
    t = t.replace('$\\to$', '->')
    t = t.replace('$-$', '-')
    t = t.replace('\\,\\%', ' %').replace('\\%', '%')
    t = re.sub(r'(?<=\d)\\,(?=\d{3})', ',', t)       # 8\,000 -> 8,000
    t = t.replace('\\ ', ' ')
    t = re.sub(r'\$(m=[0-9.]+)\$', r'\1', t)
    if TEX_LEFT.search(t):
        raise ValueError(f'LaTeX left in a text cell: {s!r} -> {t!r}')
    return t


def plain_num(s):
    """A printed numeric cell: '---' -> '', '$-$' -> '-', thousands '\\,' removed; must parse as a number."""
    t = s.strip()
    if t == '---':
        return ''
    t = t.replace('$-$', '-').replace('\\,', '')
    float(t)
    if TEX_LEFT.search(t):
        raise ValueError(f'LaTeX left in a numeric cell: {s!r}')
    return t


def yn(b):
    return 'yes' if b else 'no'


def row_cells(line):
    t = line.strip()
    if t.endswith('\\\\'):
        t = t[:-2]
    return [c.strip() for c in re.split(r'(?<!\\)&', t)]


def tabular_rows(lines, label):
    """Data rows of a tabular from its label: the lines after the first \\midrule up to \\end{tabular}, ending in \\\\."""
    i0 = next(i for i, ln in enumerate(lines) if f'\\label{{{label}}}' in ln)
    i1 = next(i for i in range(i0, len(lines)) if '\\end{tabular}' in lines[i])
    mid = next(i for i in range(i0, i1) if lines[i].strip() == '\\midrule')
    return [lines[i] for i in range(mid + 1, i1)
            if lines[i].strip() not in ('\\midrule', '\\bottomrule') and lines[i].strip().endswith('\\\\')], (i0, i1)


def csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    for r in rows:
        if len(r) != len(header):
            raise RuntimeError(f'row width {len(r)} != header width {len(header)}: {r!r}')
        w.writerow(r)
    return buf.getvalue().encode('utf-8')


# ======================================================================================================================
#  Appendix E (gen_appE.py executed, never edited)
# ======================================================================================================================
def run_gen_appE():
    """Execute gen_appE.py as a module (as verify_appE.py does), writing its LaTeX into a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix='w188b_gen_appE_')
    target = os.path.join(tmpdir, 'appE.tex')
    argv0 = list(sys.argv)
    buf = io.StringIO()
    spec = importlib.util.spec_from_file_location('gen_appE_for_supplement', os.path.join(REPO, GEN_REL))
    g = importlib.util.module_from_spec(spec)
    sys.argv = [os.path.join(REPO, GEN_REL), target]
    try:
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(g)
    finally:
        sys.argv = argv0
    with open(target, 'rb') as h:
        tex = h.read()
    os.remove(target)
    os.rmdir(tmpdir)
    return g, tex, buf.getvalue()


def build_appendix_E(g, tex):
    lines = tex.decode('utf-8').split('\n')
    line_set = set(lines)
    inv = {v: k for k, v in g.ID.items()}
    out = {}

    # ---- E.20 (rows from gen_appE.e1_row over gen_appE.E1)
    rows, no = [], 0
    for gi, (title, items) in enumerate(g.E1):
        grp = re.match(r'\(([a-d])\)', title).group(1)
        for desc, claim, src in items:
            no += 1
            ln = g.e1_row(no, desc, claim, src)
            if ln not in line_set:
                raise RuntimeError(f'E.20 row {no}: e1_row output not found in the generated LaTeX')
            c = row_cells(ln)
            if len(c) != 10 or int(c[0]) != no:
                raise RuntimeError(f'E.20 row {no}: {len(c)} cells')
            dmarks = marks_of(c[1])
            if set(dmarks) - {'b', 'Delta', 'n', 't'}:
                raise RuntimeError(f'E.20 row {no}: unexpected marks {dmarks}')
            node = re.match(r'\s*(\d+):', unmark(c[1]))
            frags = re.findall(r'([0-9.]+/[0-9.]+)\$\^\{b\}\$', c[1])
            ob_plans = '; '.join(f'{node.group(1)}: {f}' for f in frags) if frags else ''
            ev = [x.strip() for x in c[2].split('/')]
            (ea, ma), (eb, mb) = [(unmark(x), marks_of(x)) for x in ev]
            for m in ma + mb:
                if m not in ('u', 'f'):
                    raise RuntimeError(f'E.20 row {no}: unexpected eval mark {m}')
            rows.append([str(no), grp, plain_text(unmark(c[1])), claim, src,
                         ea, inv[int(ea)], yn('u' in ma), yn('f' in ma),
                         eb, inv[int(eb)], yn('u' in mb), yn('f' in mb),
                         plain_num(c[3]), plain_num(c[4]), plain_num(c[5]), plain_text(c[6]),
                         plain_num(c[7]), plain_num(c[8]), plain_text(c[9]),
                         yn('b' in dmarks), ob_plans, yn('Delta' in dmarks), yn('n' in dmarks), yn('t' in dmarks)])
    if no != 58:
        raise RuntimeError(f'E.20: {no} rows, expected 58')
    out['appendix_E/E20_comparisons.csv'] = rows

    # ---- E.21 (rows from gen_appE.e2_row over gen_appE.G)
    rows = []
    for title, items in g.G:
        grp = re.match(r'\(([a-f])\)', title).group(1)
        for cell, plan, year, cond in items:
            ln = g.e2_row(cell, plan, year, cond)
            if ln not in line_set:
                raise RuntimeError(f'E.21 {cell}: e2_row output not found in the generated LaTeX')
            c = row_cells(ln)
            if len(c) != 14 or inv[int(c[0])] != cell:
                raise RuntimeError(f'E.21 {cell}: bad row')
            pm, sm = marks_of(c[1]), marks_of(c[4])
            if set(pm) - {'b'} or set(sm) - {'f', 't', 's'} or set(marks_of(''.join(c[2:4] + c[5:]))):
                raise RuntimeError(f'E.21 {cell}: unexpected marks {pm} {sm}')
            t2 = g.T2[cell]
            rows.append([c[0], grp, cell, t2['eval key'], t2['candidate key'], plain_text(unmark(c[1])),
                         plain_num(c[2]), plain_text(c[3]), plain_text(unmark(c[4])),
                         yn('f' in sm), yn('t' in sm), yn('s' in sm), yn('b' in pm),
                         plain_num(c[5]), plain_num(c[6]), plain_num(c[7]), plain_num(c[8]), plain_num(c[9]),
                         plain_num(c[10]), plain_num(c[11]), plain_num(c[12]), plain_text(c[13])])
    if [int(r[0]) for r in rows] != list(range(1, 62)):
        raise RuntimeError('E.21: rows are not 1..61')
    out['appendix_E/E21_evaluations.csv'] = rows
    ob_by_eval = {int(r[0]): r[12] for r in rows}

    # ---- E.22 (generated table)
    trs, _ = tabular_rows(lines, 'tab:appE_neighbours')
    rows = []
    for ln in trs:
        c = row_cells(ln)
        if len(c) != 6 or marks_of(ln):
            raise RuntimeError(f'E.22: bad row {ln!r}')
        rows.append([plain_text(c[0]), plain_num(c[1]), c[2], inv[int(c[2])], plain_num(c[3]), plain_num(c[4]),
                     plain_num(c[5])])
    if len(rows) != 8:
        raise RuntimeError(f'E.22: {len(rows)} rows')
    out['appendix_E/E22_neighbours.csv'] = rows

    # ---- E.23 (generated table)
    trs, (_i0, i1_23) = tabular_rows(lines, 'tab:appE_benchmark')
    rows, arm = [], None
    for ln in trs:
        c = row_cells(ln)
        if len(c) != 8 or set(marks_of(''.join([c[0]] + c[2:]))):
            raise RuntimeError(f'E.23: bad row {ln!r}')
        arm = c[0] or arm
        sm = marks_of(c[1])
        if set(sm) - {'b'}:
            raise RuntimeError(f'E.23: unexpected marks {sm}')
        start = plain_text(unmark(c[1]))
        rows.append([plain_text(arm), start, yn('b' in sm) if start else ''] + [plain_num(x) for x in c[2:]])
    if len(rows) != 7:
        raise RuntimeError(f'E.23: {len(rows)} rows')
    out['appendix_E/E23_benchmark.csv'] = rows

    # ---- E.24 (generated table)
    trs, _ = tabular_rows(lines, 'tab:appE_gap')
    rows = []
    for ln in trs:
        c = row_cells(ln)
        if len(c) != 11 or marks_of(ln):
            raise RuntimeError(f'E.24: bad row {ln!r}')
        ev = int(c[0])
        rows.append([c[0], inv[ev], plain_text(c[1]), plain_num(c[2]), plain_text(c[3]), plain_text(c[4]),
                     ob_by_eval[ev], plain_num(c[5]), plain_num(c[6]), plain_num(c[7]), plain_num(c[8]),
                     plain_num(c[9]), plain_text(c[10])])
    if len(rows) != 9:
        raise RuntimeError(f'E.24: {len(rows)} rows')
    out['appendix_E/E24_gap_refused.csv'] = rows

    # ---- notes: the numbers gen_appE.py prints in the notes of E.20, E.22 and E.23 (its own variables)
    note20 = next(ln for ln in lines if 'In the uncertified form, row~3 has a bar of' in ln)
    note22 = next(ln for ln in lines if 'superseded first certificate of the 2025 node-5 neighbour' in ln)
    note23 = next(lines[i] for i in range(i1_23, i1_23 + 8) if '\\textit{Note:}' in lines[i])
    bp, bt, co, sup, r3, r26 = (g.bests['passive NRF'], g.bests['price-taker NRF'], g.coord, g.sup, g.r3, g.r26)
    notes = [
        ('E.20', 'row3_uncertified_form_bar_kEUR', r3['uncertified form beside: bar (k€)'], 'k EUR',
         'row 3 (evaluation 4 flagged) in the uncertified form: bar', note20),
        ('E.20', 'row3_uncertified_form_multiple', r3['uncertified form beside: ×'], '',
         'row 3 in the uncertified form: |Delta| / bar (determinate)', note20),
        ('E.20', 'row26_uncertified_form_bar_kEUR', r26['uncertified form beside: bar (k€)'], 'k EUR',
         'row 26 (evaluation 38 flagged) in the uncertified form: bar', note20),
        ('E.20', 'row26_uncertified_form_multiple', r26['uncertified form beside: ×'], '',
         'row 26 in the uncertified form: |Delta| / bar (determinate)', note20),
        ('E.22', 'eval16_superseded_delta_kEUR', sup['M gross (k€)'], 'k EUR',
         'the superseded first certificate of the 2025 node-5 neighbour (evaluation 16): Delta', note22),
        ('E.22', 'eval16_superseded_multiple', sup['×'], '', 'evaluation 16: |Delta| / threshold', note22),
        ('E.23', 'spread_passive_kEUR', bp['band (k€)'], 'k EUR', 'spread of Q over the three starts, passive NRF', note23),
        ('E.23', 'spread_price_taker_kEUR', bt['band (k€)'], 'k EUR', 'spread of Q over the three starts, price-taker NRF', note23),
        ('E.23', 'coordinated_reproducibility_band_kEUR', co['band (k€)'], 'k EUR',
         'reproducibility band of the coordinated evaluation', note23),
        ('E.23', 'sweep_passive_blocks_tn_cannot_accept', bp['sweep: blocks TN cannot accept (of 12)'], 'blocks of 12',
         'no interface rule, cold start, passive: blocks in which the TN cannot accept the DN exchange', note23),
        ('E.23', 'sweep_passive_hours', bp['sweep: hours'], 'h', 'the same, hours', note23),
        ('E.23', 'sweep_passive_failing_blocks', bp['sweep: failing blocks'], '', 'the same, blocks ("; "-separated)', note23),
        ('E.23', 'sweep_price_taker_blocks_tn_cannot_accept', bt['sweep: blocks TN cannot accept (of 12)'], 'blocks of 12',
         'no interface rule, cold start, price-taker: blocks in which the TN cannot accept the DN exchange', note23),
        ('E.23', 'sweep_price_taker_hours', bt['sweep: hours'], 'h', 'the same, hours', note23),
        ('E.23', 'sweep_price_taker_failing_blocks', bt['sweep: failing blocks'], '', 'the same, blocks ("; "-separated)', note23),
    ]
    rows = []
    for t, item, val, unit, desc, note in notes:
        shown = val.replace('; ', ', ') if item == 'sweep_price_taker_failing_blocks' else val
        if not val or shown not in note:
            raise RuntimeError(f'note value {t} {item} = {val!r} not printed in its note')
        rows.append([t, item, val, unit, desc])
    out['appendix_E/E_table_notes.csv'] = rows

    # ---- group titles (verbatim LaTeX source)
    rows = [['E.20', re.match(r'\(([a-d])\)', t).group(1), t] for t, _ in g.E1]
    rows += [['E.21', re.match(r'\(([a-f])\)', t).group(1), t] for t, _ in g.G]
    out['appendix_E/E20_E21_groups.csv'] = rows
    return out


# ======================================================================================================================
#  Appendices B-D (case files)
# ======================================================================================================================
def _hours_of(df, first_hour_col):
    cols = list(df.columns)[first_hour_col:]
    if [int(c) for c in cols] != list(range(24)):
        raise RuntimeError(f'hour columns are not 0..23: {cols}')
    return cols


def _json(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


def build_appendices_B_D():
    out, sources, checks = {}, set(), {}

    # ---- B: market prices
    mk = SRP._read_market_base_profiles(os.path.join(REPO, MARKET_XLSX))
    sources.add(MARKET_XLSX)
    rows = []
    for price, key in (('energy', 'energy'), ('flexibility', 'flexibility')):
        df = mk[key]
        if list(df.columns)[:2] != ['Season', 'Scenario']:
            raise RuntimeError(f'market sheet {key}: columns {list(df.columns)[:2]}')
        hc = _hours_of(df, 2)
        for r in df.itertuples(index=False):
            rr = list(r)
            rows.append([price, fmt(rr[0]), fmt(rr[1])] + [fmt(v) for v in rr[2:2 + len(hc)]])
    out['appendices_B_D/B_market_prices.csv'] = rows

    # ---- network JSON: read every case year; the network tables must be identical across years
    nets = {n: {y: _json(net_json(n, y)) for y in ALL_CASE_YEARS} for n in [TN] + [a for a, _ in ADNS]}
    for n in nets:
        for y in ALL_CASE_YEARS:
            sources.add(net_json(n, y))
    for n in nets:
        base = nets[n][2025]
        for y in ALL_CASE_YEARS:
            d = nets[n][y]
            for k in ('nodes', 'loads', 'lines') + (('transformers',) if n != TN else ()):
                if d[k] != base[k]:
                    raise RuntimeError(f'{n} {y}: {k} differs from 2025')
    checks['network_tables_identical_across_case_years'] = {'years': list(ALL_CASE_YEARS), 'holds': True}

    # ---- C: TN RES generators and profiles
    def res_rows(n):
        ids = [g['gen_id'] for g in nets[n][2025]['generators'] if g['type'] in GEN_TYPE_LABEL]
        rr = []
        for gid in ids:
            per_year = []
            for y in PRINTED_YEARS:
                gg = [g for g in nets[n][y]['generators'] if g['gen_id'] == gid]
                if len(gg) != 1:
                    raise RuntimeError(f'{n} {y}: generator {gid} found {len(gg)} times')
                per_year.append(gg[0])
            if len({(g['bus'], g['type']) for g in per_year}) != 1:
                raise RuntimeError(f'{n}: generator {gid} changes bus or type across years')
            g0 = per_year[0]
            rr.append([fmt(gid), fmt(g0['bus']), GEN_TYPE_LABEL[g0['type']]] + [fmt(g['Pmax']) for g in per_year])
        return rr
    out['appendices_B_D/C_tn_res_generators.csv'] = res_rows(TN)

    opd = {n: ND._read_network_base_profiles(os.path.join(REPO, net_xlsx(n))) for n in [TN] + [a for a, _ in ADNS]}
    for n in opd:
        sources.add(net_xlsx(n))

    def prof_rows(df, n_keys):
        hc = _hours_of(df, n_keys)
        return [[fmt(v) for v in list(r)[:n_keys]] + [fmt(v) for v in list(r)[n_keys:n_keys + len(hc)]]
                for r in df.itertuples(index=False)]
    out['appendices_B_D/C_tn_res_profiles.csv'] = prof_rows(opd[TN]['pg'], 3)

    # ---- D: the IEEE 33-bus tables (from case33_1; the other two ADNs checked equal, ratings / TN connection per ADN)
    a1 = nets['case33_1'][2025]
    for n, _k in ADNS[1:]:
        d = nets[n][2025]
        for k in ('nodes', 'loads', 'lines'):
            if d[k] != a1[k]:
                raise RuntimeError(f'{n}: {k} differs from case33_1')
        t0, t1 = d['transformers'], a1['transformers']
        if len(t0) != 1 or {k: v for k, v in t0[0].items() if k != 'rating'} != {k: v for k, v in t1[0].items() if k != 'rating'}:
            raise RuntimeError(f'{n}: transformer differs from case33_1 beyond its rating')
    out['appendices_B_D/D_ieee33_buses.csv'] = [
        [fmt(b['bus_i']), fmt(b['type']), BUS_TYPE_LABEL[b['type']], fmt(b['Gs']), fmt(b['Bs']), fmt(b['baseKV']),
         fmt(b['Vmax']), fmt(b['Vmin'])] for b in a1['nodes']]
    br = [('transformer', x) for x in a1['transformers']] + [('line', x) for x in a1['lines']]
    br.sort(key=lambda t: t[1]['branch_id'])
    if [x['branch_id'] for _e, x in br] != list(range(1, len(br) + 1)):
        raise RuntimeError('branch ids are not 1..N')
    rating = {}
    for n, k in ADNS:
        d = nets[n][2025]
        rating[k] = {x['branch_id']: x['rating'] for x in d['transformers'] + d['lines']}
    out['appendices_B_D/D_ieee33_branches.csv'] = [
        [fmt(x['branch_id']), e, fmt(x['fbus']), fmt(x['tbus']), fmt(x['r']), fmt(x['x']), fmt(x['b'])]
        + [fmt(rating[k][x['branch_id']]) for _n, k in ADNS] for e, x in br]
    rows = []
    for n, k in ADNS:
        ref = [g for g in nets[n][2025]['generators'] if g['type'] == 'REF']
        if len(ref) != 1:
            raise RuntimeError(f'{n}: {len(ref)} REF generators')
        g0 = ref[0]
        rows.append([str(k), n, fmt(g0['gen_id']), fmt(g0['bus']), fmt(g0['Pmax']), fmt(g0['Pmin']), fmt(g0['Qmax']),
                     fmt(g0['Qmin']), fmt(g0['Vg'])])
    out['appendices_B_D/D_ieee33_tn_connection.csv'] = rows
    chars = {}
    for n, k in ADNS:
        ch = opd[n]['characterization']
        if list(ch.columns)[:4] != ['LoadID', 'Pc', 'Qc', 'FL Ratio']:
            raise RuntimeError(f'{n}: Characterization columns {list(ch.columns)[:4]}')
        chars[k] = {int(r[0]): r for r in ch.itertuples(index=False)}   # r[0] LoadID, r[1] Pc, r[2] Qc, r[3] FL Ratio
    fl = {k: {lid: float(r[3]) for lid, r in chars[k].items()} for k in chars}
    if not (fl[5] == fl[7] == fl[9]):
        raise RuntimeError('FL Ratio differs between the ADNs')
    rows = []
    for ld in a1['loads']:
        lid = ld['load_id']
        r1 = chars[5][lid]
        row = [fmt(lid), fmt(ld['bus']), fmt(r1[3])]
        for _n, k in ADNS:
            row += [fmt(chars[k][lid][1]), fmt(chars[k][lid][2])]
        rows.append(row)
    if sorted(int(r[0]) for r in rows) != sorted(chars[5]):
        raise RuntimeError('loads in the JSON and in Characterization differ')
    out['appendices_B_D/D_ieee33_loads.csv'] = rows

    rows = []
    for n, k in ADNS:
        rows += [[str(k), n] + r for r in res_rows(n)]
    out['appendices_B_D/D_adn_res_generators.csv'] = rows
    rows = []
    for n, k in ADNS:
        rows += [[str(k), n] + r for r in prof_rows(opd[n]['pg'], 3)]
    out['appendices_B_D/D_adn_res_profiles.csv'] = rows
    rows = []
    for n, k in ADNS:
        for q, key in (('Pc', 'pc'), ('Qc', 'qc')):
            df = opd[n][key]
            if list(df.columns)[:3] != ['LoadID', 'Season', 'Scenario']:
                raise RuntimeError(f'{n} {key}: columns')
            rows += [[str(k), n, q] + r for r in prof_rows(df, 3)]
    out['appendices_B_D/D_adn_load_profiles.csv'] = rows
    rows = []
    for n, k in ADNS:
        df = opd[n]['flex']
        if list(df.columns)[:3] != ['Type', 'Season', 'Scenario']:
            raise RuntimeError(f'{n} flex: columns')
        rows += [[str(k), n] + r for r in prof_rows(df, 3)]
    out['appendices_B_D/D_adn_flexibility_profiles.csv'] = rows
    for f in CONFIG_FILES:
        sources.add(f)
    for f in CONFIG_FILES:
        cfg = _json(f)
        got = [(d['name'], d['connection_node_id']) for d in cfg['DistributionNetworks']]
        if got != [(n, k) for n, k in ADNS] or cfg['TransmissionNetwork']['name'] != TN \
                or cfg['MarketData'] != os.path.basename(MARKET_XLSX):
            raise RuntimeError(f'{f}: network / market file mapping differs from ADNS / TN / MARKET_XLSX')
    checks['adn_to_tn_node_mapping_from_configs'] = {'files': list(CONFIG_FILES), 'mapping': [list(x) for x in ADNS],
                                                     'holds': True}
    return out, sorted(sources), checks


# ======================================================================================================================
#  README placeholder (generated from COLUMN_SPECS and from the code's constants)
# ======================================================================================================================
def readme(g, version):
    spec6 = _json(SPEC_V6_REL)
    pmax = sorted({c['declaration']['settling_rule']['p_max'] for c in spec6['cells'].values()})
    if len(pmax) != 1:
        raise RuntimeError(f'P_MAX not unique in the v6 spec: {pmax}')
    consts = SC6.constants(pmax[0])
    fz = _json(FZ_REL)
    rules = sorted({r['rule'] for r in g.T1.values()} | {r['rule'] for r in g.T10.values()})
    L = []
    P = L.append
    P(f'# SRP1 supplement, version {version} -- README PLACEHOLDER')
    P('')
    P('> **[EXPERT PROSE PLACEHOLDER -- the expert writes this README.]** This file is generated by '
      '`manuscript_review/gen_supplement.py`. The column definitions below are generated from the generator\'s own '
      'column specifications; the values in the marked placeholders are filled from the code and the frozen tables '
      'where they exist. Nothing below is typed by hand except the column descriptions in the generator.')
    P('')
    P('## [PLACEHOLDER: tau]')
    P('')
    P(f'- `settling_criterion_v6.TAU` = {SC6.TAU!r} EUR (formula: {consts["TAU"]["formula"]}); printed 4,539.07 EUR.')
    P(f'- frozen tables `constants.TAU` = {fz["constants"]["TAU"]!r} EUR; `constants.TWO_TAU` = '
      f'{fz["constants"]["TWO_TAU"]!r} EUR.')
    P('')
    P('## [PLACEHOLDER: objective convention]')
    P('')
    P('- Q: gross operational cost, contracted interface settlement excluded.')
    P(f'- frozen tables `objective_convention`: "{fz["objective_convention"]}"')
    P('')
    P('## [PLACEHOLDER: certification rule parameters]')
    P('')
    P(f'- Stop rule: `settling_criterion_v6` (VERSION {SC6.VERSION}); P_MAX = {pmax[0]} cycles, read from '
      f'`{SPEC_V6_REL}` (every cell declares `settling_rule.p_max` = {pmax[0]}).')
    def _numeric(x):
        if isinstance(x, bool):
            return False
        if isinstance(x, (int, float)):
            return True
        return isinstance(x, dict) and bool(x) and all(_numeric(y) for y in x.values())
    other = []
    for k in sorted(consts):
        v = consts[k]
        val = v.get('value') if isinstance(v, dict) else v
        form = v.get('formula') if isinstance(v, dict) else ''
        if not _numeric(val):
            other.append(k)
            continue
        if not isinstance(form, str):
            form = f'sources: settling_criterion_v6.constants({pmax[0]})["{k}"]["formula"]'
        P(f'- `{k}` = `{json.dumps(val, sort_keys=True)}` -- {form}')
    P(f'- Non-numeric entries of `settling_criterion_v6.constants({pmax[0]})` (rule definitions, see the code): '
      f'{", ".join(other)}.')
    P(f'- Determinacy rules as recorded in the export (`rule` column of T1 / T10): {"; ".join(rules)}.')
    P('')
    P('## Files and columns')
    P('')
    P('Empty CSV cell: not applicable, not recorded, or (Appendix E) printed "---". Booleans: yes / no. Appendix E '
      'amounts are in k EUR as printed (rounded per export_W160/README.md); Appendix B-D values are at the full '
      'precision of the case files.')
    for f in sorted(COLUMN_SPECS):
        s = COLUMN_SPECS[f]
        P('')
        P(f'### `{f}`')
        P('')
        P(f'{s["table"]}. Source: {s["source"]}.')
        P('')
        P('| column | unit | definition |')
        P('|---|---|---|')
        for name, unit, dfn in s['columns']:
            P(f'| `{name}` | {unit} | {dfn} |')
    P('')
    P('## [EXPERT PROSE PLACEHOLDER: figures of Appendices B-D]')
    P('')
    P('The Appendix B-D figures plot the mean and standard deviation over synthetic scenarios that production samples '
      'from these base profiles (network_data.generate_*_profiles, shared_resources_planning._generate_market_price_'
      'scenarios); the supplement carries the base profiles of the case files, not the plotted curves.')
    P('')
    return ('\n'.join(L) + '\n').encode('utf-8')


# ======================================================================================================================
#  zip
# ======================================================================================================================
def build_zip(members):
    """members: {name: bytes}. Deterministic: sorted names, fixed date, fixed mode, ZIP_STORED."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as z:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = ZIP_MODE
            zi.create_system = 3
            z.writestr(zi, members[name])
    return buf.getvalue()


# ======================================================================================================================
#  main
# ======================================================================================================================
def resolve_out_dir(args):
    if args.out_dir:
        od = os.path.abspath(args.out_dir)
        if od == REPO or od.startswith(REPO + os.sep):
            raise SystemExit('refused: --out-dir must be outside the repository (scratch copies only)')
        return od, 1, None, False
    canon = os.path.join(REPO, CANONICAL_DIR)
    if not args.force_new_version:
        return canon, 1, None, True
    recs = []
    for d in sorted(os.listdir(os.path.join(REPO, MR))):
        full = os.path.join(REPO, MR, d)
        if d == 'supplement' or re.fullmatch(r'supplement_v\d+', d):
            for f in os.listdir(full) if os.path.isdir(full) else []:
                m = re.fullmatch(r'supplement_v(\d+)_record\.json', f)
                if m:
                    recs.append((int(m.group(1)), os.path.join(full, f)))
    if not recs:
        raise SystemExit('refused: --force-new-version but no earlier version record found')
    prev_v, prev_rec = max(recs)
    v = prev_v + 1
    with open(prev_rec, encoding='utf-8') as h:
        prev = json.load(h)
    return os.path.join(REPO, MR, f'supplement_v{v}'), v, {'version': prev_v, 'record': os.path.relpath(prev_rec, REPO),
                                                          'zip': prev['zip']['name'], 'zip_sha256': prev['zip']['sha256']}, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', help='scratch output directory OUTSIDE the repository (determinism checks)')
    ap.add_argument('--force-new-version', action='store_true',
                    help='write the next version into manuscript_review/supplement_v<N>/ (never overwrites)')
    args = ap.parse_args()
    if args.out_dir and args.force_new_version:
        raise SystemExit('refused: --out-dir and --force-new-version are exclusive')
    copies = other_copies('gen_supplement.py')
    if copies:
        _log(f'refused: another copy is running: {copies}')
        return 1
    out_dir, version, predecessor, canonical = resolve_out_dir(args)
    if os.path.exists(out_dir):
        extra = sorted(set(os.listdir(out_dir)) - {LAUNCH_LOG})
        if extra:
            _log(f'refused: {out_dir} is not empty ({extra[:5]}); an existing supplement is never overwritten '
                 '(use --force-new-version for a new version)')
            return 1
    inputs = [GEN_REL, APPX_REL, FZ_REL, SPEC_V6_REL, 'settling_criterion_v6.py', 'settling_criterion_v5.py',
              'shared_resources_planning.py', 'network_data.py'] + [os.path.join(EXPORT_DIR, f) for f in EXPORT_FILES]
    if canonical:
        dirty = [f for f in inputs + [SCRIPT_REL] if not _committed_clean(f)]
        if dirty:
            _log(f'refused: the canonical run needs these committed and clean: {dirty}')
            return 1
    _log(f'out_dir = {out_dir}; version {version}; predecessor {predecessor}')

    g, appe_tex, gen_stdout = run_gen_appE()
    _log(f'gen_appE.py stdout: {gen_stdout.strip()!r}')
    if not gen_stdout.startswith('ok 58 61'):
        raise RuntimeError(f'gen_appE.py stdout {gen_stdout!r}')
    E = build_appendix_E(g, appe_tex)
    BD, bd_sources, bd_checks = build_appendices_B_D()
    tables = {**E, **BD}
    if set(tables) != set(COLUMN_SPECS):
        raise RuntimeError(f'tables vs specs: {set(tables) ^ set(COLUMN_SPECS)}')
    members = {}
    for name in sorted(tables):
        header = [c[0] for c in COLUMN_SPECS[name]['columns']]
        members[name] = csv_bytes(header, tables[name])
        _log(f'{name}: {len(tables[name])} rows x {len(header)} columns')
    members['README_PLACEHOLDER.md'] = readme(g, version)
    sums = ''.join(f'{_sha_bytes(members[n])}  {n}\n' for n in sorted(members))
    members['SHA256SUMS.txt'] = sums.encode('utf-8')
    zbytes = build_zip(members)
    zsha = _sha_bytes(zbytes)
    zname = f'{NAME_STEM}_v{version}_{zsha[:8]}.zip'
    record = {
        'stage': 'W188 (b) supplement generator (Addendum 89 ruling 8)',
        'version': version, 'predecessor': predecessor,
        'zip': {'name': zname, 'sha256': zsha, 'bytes': len(zbytes), 'compression': 'ZIP_STORED',
                'entry_date_time': list(ZIP_DATE), 'entry_mode': oct(ZIP_MODE >> 16), 'entries_sorted': True},
        'members': [{'name': n, 'sha256': _sha_bytes(members[n]), 'bytes': len(members[n]),
                     'rows': len(tables[n]) if n in tables else None,
                     'columns': len(COLUMN_SPECS[n]['columns']) if n in COLUMN_SPECS else None} for n in sorted(members)],
        'appendix_E': {'gen_appE_stdout': gen_stdout.strip(), 'generated_tex_sha256': _sha_bytes(appe_tex),
                       'generated_equals_committed_appendix': _sha_bytes(appe_tex) == _sha(APPX_REL)},
        'appendices_B_D': {'case_files_read': bd_sources, 'checks': bd_checks},
        'inputs_sha256': {f: _sha(f) for f in sorted(set(inputs + bd_sources))},
        'generator': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': _committed_clean(SCRIPT_REL)},
        'git_head': _git('rev-parse', 'HEAD'),
        'environment': {'python': sys.version.split()[0], 'executable': sys.executable, 'platform': platform.platform(),
                        'numpy': np.__version__, 'pandas': __import__('pandas').__version__,
                        'openpyxl': __import__('openpyxl').__version__},
        'out_dir': os.path.relpath(out_dir, REPO) if canonical else out_dir, 'canonical': canonical,
    }
    fails = GUARD.verify(0)
    record['guards'] = {'solve_profile_guard': dict(GUARD.counts), 'verify_0': fails, 'pickle_calls': dict(PICKLE_COUNTS)}
    if fails or any(PICKLE_COUNTS.values()):
        _log(f'guard fault: {fails} {PICKLE_COUNTS}')
        return 1
    if not record['appendix_E']['generated_equals_committed_appendix']:
        _log('NOTE: the gen_appE.py output differs from the committed appendixE_evaluation_record.tex (recorded)')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, zname), 'xb') as h:
        h.write(zbytes)
    with open(os.path.join(out_dir, f'supplement_v{version}_record.json'), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(record, indent=1, sort_keys=True) + '\n')
    _log(f'wrote {zname} ({len(zbytes)} bytes) sha256 {zsha}')
    _log(f'guards: {GUARD.counts} pickle {PICKLE_COUNTS}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
