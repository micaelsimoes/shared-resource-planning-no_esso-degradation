#!/usr/bin/env python3
"""Generate Appendix E (Evaluation Record) from the W160 export CSVs.
Every number printed in the tables is read from export_W160/*.csv; nothing is typed by hand.
"""
import csv, sys, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export_W160')
def rd(n): return list(csv.DictReader(open(os.path.join(D, n), encoding='utf-8')))
T1 = {r['claim']: r for r in rd('T1.csv')}
T2 = {r['cell']: r for r in rd('T2.csv')}
T5 = {r['cell']: r for r in rd('T5.csv')}
T6 = rd('T6.csv')
T9 = {r['cell']: r for r in rd('T9.csv')}
T10 = {r['claim']: r for r in rd('T10.csv')}
assert len(T1) == 60 and len(T2) == 61, (len(T1), len(T2))

X0 = r'$\boldsymbol{x}=\boldsymbol{0}$'
# ---------------------------------------------------------------- evaluations (Table E.2)
# (cell, plan, year, condition)
G = []  # groups: (title, [(cell, plan, year, condition)])
G.append((r'(a) Baseline price: the plan without storage and single-node plans in 2025 (Tables~\ref{tab:res_value_ladder} and~\ref{tab:res_breakeven}, Fig.~\ref{fig:res_breakeven})', [
    ('ref:7aa017f0', X0, '---', 'base'),
    ('ref:bd504ecf', '7: 0.25/1', '2025', 'base'),
    ('d_c52e1670', '7: 0.5/2', '2025', 'base'),
    ('d_4a82a64a', '7: 0.75/3', '2025', 'base'),
    ('d_3632b0ae', '7: 1/4', '2025', 'base'),
    ('d_36686489', '7: 1.25/5', '2025', 'base'),
    ('d_d3709599', '7: 0.5/1', '2025', 'base'),
    ('d_a12d95a2', '7: 1/2', '2025', 'base'),
    ('d_f759dd48', '7: 1.5/3', '2025', 'base'),
    ('d_c7fee8be', '7: 2/4', '2025', 'base'),
    ('d_9246ed01', '7: 2.5/5', '2025', 'base'),
    ('b_2a0ba8b2', '5: 0.25/1', '2025', 'base'),
    ('b_0dd237f0', '9: 0.25/1', '2025', 'base'),
    ('b_4649234b', '9: 0.75/3', '2025', 'base'),
]))
G.append((r'(b) Neighbours of the plan without storage (Tables~\ref{tab:res_value_ladder} and~\ref{tab:appE_neighbours})', [
    ('pb_y2025_n5_v6', '5: 0.25/0.5', '2025', 'base'),
    ('pb_y2025_n5', '5: 0.25/0.5', '2025', 'base'),
    ('pb_y2025_n7', '7: 0.25/0.5', '2025', 'base'),
    ('pb_y2025_n9', '9: 0.25/0.5', '2025', 'base'),
    ('pb_y2030_n5', '5: 0.25/0.5', '2030', 'base'),
    ('pb_y2030_n7', '7: 0.25/0.5', '2030', 'base'),
    ('pb_y2030_n9', '9: 0.25/0.5', '2030', 'base'),
    ('c_156ce2d1', '5, 9: 0.25/0.5 each', '2025', 'base'),
    ('c_6597a79d', '5, 7: 0.25/0.5 each', '2030', 'base'),
]))
G.append((r'(c) Investment year (Table~\ref{tab:res_year_ladder})', [
    ('yl_y2030', '7: 0.25/1', '2030', 'base'),
    ('yl_y2035', '7: 0.25/1', '2035', 'base'),
    ('g_37b5c499', '7: 0.5/1', '2030', 'base'),
    ('g_47dce43c', '7: 0.5/1', '2035', 'base'),
    ('g_48749148', '7: 0.5/2', '2030', 'base'),
    ('g_9abf31d4', '7: 0.5/2', '2035', 'base'),
]))
G.append((r'(d) Flexibility price (Tables~\ref{tab:res_flex_ladder} and~\ref{tab:res_energy_ladder_m2})', [
    ('h_aa8a76d7', X0, '---', '$m=1.5$'),
    ('h_f9eae48f', '7: 0.25/1', '2025', '$m=1.5$'),
    ('h_x0_m175', X0, '---', '$m=1.75$'),
    ('h_unit_m175', '7: 0.25/1', '2025', '$m=1.75$'),
    ('h_50dea31c', X0, '---', '$m=2$'),
    ('h_74eda68d', '7: 0.25/1', '2025', '$m=2$'),
    ('i_5a6a88b4', '7: 0.5/2', '2025', '$m=2$'),
    ('j_f3aa335e', '7: 0.75/3', '2025', '$m=2$'),
    ('j_a11d7966', '7: 1/4', '2025', '$m=2$'),
    ('j_5f3cccb4', '7: 1.25/5', '2025', '$m=2$'),
]))
G.append((r'(e) Twice the base flexibility price: the plan returned by the search and its neighbours (Subsection~\ref{subsec:res_breakeven})', [
    ('ref:5ca4f86c', '5: 0.25/0.5; 7: 1/3.5', '2030', '$m=2$'),
    ('ref:e28de4ac', '5: 0.25/1; 7: 1/3', '2030', '$m=2$'),
    ('l_0ee93aca', '5: 0.25/0.5; 7: 1/3', '2030', '$m=2$'),
    ('l_195156fa', '5: 0.25/0.5; 7: 0.75/3', '2030', '$m=2$'),
    ('l_2ab0ce2d', '5: 0.25/0.5; 7: 1.25/3', '2030', '$m=2$'),
    ('l_b2251bc5', '5: 0.25/0.5; 7: 0.75/3; 9: 0.25/0.5', '2030', '$m=2$'),
    ('l_76c78064', '7: 1/3', '2030', '$m=2$'),
    ('l_8e4c220e', '7: 0.75/3', '2030', '$m=2$'),
    ('l_e1da0984', '7: 1/3.5', '2030', '$m=2$'),
    ('l_7b199ef9', '7: 1/4', '2030', '$m=2$'),
    ('l_45aa25a6', '7: 1/3; 9: 0.25/0.5', '2030', '$m=2$'),
    ('l_7c455554', '7: 1/3.5; 9: 0.25/0.5', '2030', '$m=2$'),
    ('l_7db09f6c', '7: 0.75/3; 9: 0.25/0.5', '2030', '$m=2$'),
    ('l_df1a5525', '5: 0.25/0.5; 7: 0.75/2.5', '2030', '$m=2$'),
]))
G.append((r'(f) Ageing calibrations of the reference unit (Table~\ref{tab:res_ageing}, Fig.~\ref{fig:res_ageing})', [
    ('e_no_ageing', '7: 0.25/1', '2025', 'no ageing'),
    ('e_c2', '7: 0.25/1', '2025', 'cycling only'),
    ('e_c2_calfade', '7: 0.25/1', '2025', 'baseline'),
    ('e_c4', '7: 0.25/1', '2025', 'datasheet'),
    ('e_c3_unit', '7: 0.25/1', '2025', 'retention 50\\,\\%'),
    ('e_c3_midblock', '7: 0.25/1', '2025', 'ret.\\ 50\\,\\%, mid-block'),
    ('e_soh050', '7: 0.25/1', '2025', 'baseline, 0.50 floor'),
    ('g070_neutrality', '7: 0.25/1', '2025', 'baseline, 0.70 floor'),
]))
ID = {}
n = 0
for _, rows in G:
    for c, *_ in rows:
        n += 1; ID[c] = n
assert n == 61 and set(ID) == set(T2), set(T2) ^ set(ID)

def neg(s):
    """typeset a number string with a math minus"""
    s = s.strip()
    if s.startswith('-'):
        return '$-$' + s[1:]
    return s
def ks(s):  # thousands in k EUR for Q column of T6
    v = s.strip()
    if not v: return '---'
    ip, _, dp = v.partition('.')
    sign = ''
    if ip.startswith('-'): sign, ip = '$-$', ip[1:]
    ip = '{:,}'.format(int(ip)).replace(',', '\\,')
    return sign + ip + ('.' + dp if dp else '')
def d(s): return s.strip() if s.strip() else '---'

SUPERSEDED = {'pb_y2025_n5'}

def status(r):
    if r['cell'] in SUPERSEDED:
        return 'superseded'
    if r['status'] == 'certified':
        b = r['branch']
        s = {'oscillatory': 'cert., osc.', 'monotone': 'cert., mon.', '': 'cert.'}[b]
    else:
        cause = r['cause (uncertified)']
        if cause.startswith('growth test'):
            s = 'growth test'
        else:
            s = {'gap clause': 'gap clause', 'lapse reset': 'lapse reset'}[cause]
    return s

def rule_col(c, r):
    spec = r['certifying spec']
    if c in SUPERSEDED:
        return '---'
    if c in ('ref:7aa017f0', 'ref:bd504ecf', 'ref:5ca4f86c', 'ref:e28de4ac'):
        return 'record'
    if spec in ('v6', 'ext v3 (rule v6)', 'A64 v1 (rule v6)') or (spec == '' and r['status'] == 'uncertified'):
        return 'run'
    if spec in ('v4', 'v5', 'W118 r2 (rule v2)', 'v6 (v6 from records)'):
        return 'record'
    raise ValueError((c, spec))

FLAG = {'d_4a82a64a', 'j_a11d7966'}
TWIN = {'e_c2_calfade', 'g070_neutrality'}

def e2_row(c, plan, year, cond):
    r = T2[c]
    st = status(r)
    mark = ''
    if c in FLAG: mark = '$^{f}$'
    if c in TWIN: mark = '$^{t}$'
    if c in SUPERSEDED: mark = '$^{s}$'
    cert = r['status'] == 'certified'
    k0 = d(r['k0'])
    kstar = d(r['k* / end'])
    rng = d(r['range/τ']) if cert else '---'
    band = d(r['band (k€)']) if cert else '---'
    gap = neg(d(r['gap (k€)'])) if not cert else '---'
    slack = d(r['slack (k€)']) if not cert else '---'
    nc = d(r['non-clean cycles after N'])
    rep = d(r['replay bitwise through'])
    return f"{ID[c]} & {plan} & {year} & {cond} & {st}{mark} & {k0} & {kstar} & {rng} & {band} & {gap} & {slack} & {nc} & {rep} & {rule_col(c, r)} \\\\"

# ---------------------------------------------------------------- comparisons (Table E.1)
VERD = {'determinate': 'det.', 'within resolution': 'w.r.', 'within the uncertified bar': 'w.b.'}
def unc(c): return T2[c]['status'] == 'uncertified'
def ev(c):
    s = str(ID[c])
    if unc(c): s += '$^{u}$'
    if c in FLAG: s += '$^{f}$'
    return s
def cell_of(x):  # T1 cells: 'ref:xxx' keep, else as is
    return x
def e1_row(no, desc, claim, src='T1'):
    if src == 'T1':
        r = T1[claim]
        a, b = r['ref cell'], r['other cell']
        g, thr, xg, vg = r['d gross (k€)'], r['threshold / bar (k€)'], r['× gross'], r['verdict (gross)']
        dn, xn, vn = r['d net (k€)'], r['× net'], r['verdict (net)']
        net = f"{neg(dn)} & {xn} & {VERD[vn]}"
    else:
        r = T10[claim]
        a, b = r['ref'], r['other']
        a = {'7aa017f0': 'ref:7aa017f0', 'bd504ecf': 'ref:bd504ecf'}.get(a, a)
        g, thr, xg, vg = r['d gross (k€)'], r['threshold (k€)'], r['×'], r['verdict']
        net = '--- & --- & ---'
    return f"{no} & {desc} & {ev(a)} / {ev(b)} & {neg(g)} & {thr} & {xg} & {VERD[vg]} & {net} \\\\"

E1 = []  # (group title, [(desc, claim, src)])
E1.append((r'(a) Plans against the plan without storage at the baseline price: $\Delta = F(\boldsymbol{x}) - F(\boldsymbol{0})$; positive: the plan without storage costs less', [
    ('7: 0.25/1 (reference unit)', 'B:n7_4h_e1', 'T1'),
    ('7: 0.5/2', 'B:n7_4h_e2', 'T1'),
    ('7: 0.75/3', 'B:n7_4h_e3', 'T1'),
    ('7: 1/4', 'B:n7_4h_e4', 'T1'),
    ('7: 1.25/5', 'B:n7_4h_e5', 'T1'),
    ('7: 0.5/1', 'B:n7_2h_e1', 'T1'),
    ('7: 1/2', 'B:n7_2h_e2', 'T1'),
    ('7: 1.5/3', 'B:n7_2h_e3', 'T1'),
    ('7: 2/4', 'B:n7_2h_e4', 'T1'),
    ('7: 2.5/5', 'B:n7_2h_e5', 'T1'),
    ('5: 0.25/1', 'B:n5_4h_e1', 'T1'),
    ('9: 0.25/1', 'B:n9_4h_e1', 'T1'),
    ('9: 0.75/3', 'B:n9_4h_e3', 'T1'),
    ('5: 0.25/0.5', 'C:y2025__n5_p0.25_e0.5', 'T1'),
    ('5, 9: 0.25/0.5 each', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', 'T1'),
    ('5, 7: 0.25/0.5 each, 2030', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'T1'),
    ('7: 0.5/1, 2030', 'G:n7_2h_e1_y2030:gross', 'T1'),
    ('7: 0.5/1, 2035', 'G:n7_2h_e1_y2035:gross', 'T1'),
    ('7: 0.5/2, 2030', 'G:n7_4h_e2_y2030:gross', 'T1'),
    ('7: 0.5/2, 2035', 'G:n7_4h_e2_y2035:gross', 'T1'),
]))
E1.append((r'(b) Flexibility price $m$, plans in 2025 against the plan without storage at the same price: $\Delta = \text{value} - I$; positive: the plan pays', [
    ('7: 0.25/1, $m=1.5$', 'H:m1.5:value_minus_I', 'T1'),
    ('7: 0.25/1, $m=1.75$', 'H:m1.75:value_minus_I', 'T10'),
    ('7: 0.25/1, $m=2$', 'H:m2:value_minus_I', 'T1'),
    ('7: 0.5/2, $m=2$', 'I:m2:e2_value_minus_I', 'T1'),
    ('7: 0.75/3, $m=2$', 'J:e3_value_minus_I', 'T1'),
    ('7: 1/4, $m=2$', 'J:e4_value_minus_I', 'T1'),
    ('7: 1.25/5, $m=2$', 'J:e5_value_minus_I', 'T1'),
    (r'7: 0.25/1 $\to$ 0.5/2, $m=2$$^{\Delta}$', 'I:m2:second_MWh', 'T1'),
    (r'7: 0.5/2 $\to$ 0.75/3, $m=2$$^{\Delta}$', 'J:e2_to_e3', 'T1'),
    (r'7: 0.75/3 $\to$ 1/4, $m=2$$^{\Delta}$', 'J:e3_to_e4', 'T1'),
    (r'7: 1/4 $\to$ 1.25/5, $m=2$$^{\Delta}$', 'J:e4_to_e5', 'T1'),
]))
E1.append((r'(c) Twice the base flexibility price, 2030 unless stated: neighbours against the plan returned by the search (5: 0.25/0.5; 7: 1/3.5, evaluation~40), $\Delta = F(\text{neighbour}) - F(\text{returned plan})$; positive: the returned plan costs less', [
    ('7: 0.75/3, 2025', 'L:y2025__n7_p0.75_e3', 'T1'),
    ('5: 0.25/0.5; 7: 0.75/3', 'L:y2030__n5_p0.25_e0.5__n7_p0.75_e3', 'T1'),
    ('5: 0.25/0.5; 7: 1/3', 'L:y2030__n5_p0.25_e0.5__n7_p1_e3', 'T1'),
    ('5: 0.25/0.5; 7: 1.25/3', 'L:y2030__n5_p0.25_e0.5__n7_p1.25_e3', 'T1'),
    ('5: 0.25/1; 7: 1/3', 'L:y2030__n5_p0.25_e1__n7_p1_e3', 'T1'),
    ('5: 0.25/0.5; 7: 0.75/3; 9: 0.25/0.5', 'L:y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5', 'T1'),
    ('7: 0.75/3', 'L:y2030__n7_p0.75_e3', 'T1'),
    ('7: 1/3', 'L:y2030__n7_p1_e3', 'T1'),
    ('7: 1/3.5', 'L:y2030__n7_p1_e3.5', 'T1'),
    ('7: 1/4', 'L:y2030__n7_p1_e4', 'T1'),
    ('7: 0.75/3; 9: 0.25/0.5', 'L:y2030__n7_p0.75_e3__n9_p0.25_e0.5', 'T1'),
    ('7: 1/3; 9: 0.25/0.5', 'L:y2030__n7_p1_e3__n9_p0.25_e0.5', 'T1'),
    ('7: 1/3.5; 9: 0.25/0.5', 'L:y2030__n7_p1_e3.5__n9_p0.25_e0.5', 'T1'),
    ('5: 0.25/0.5; 7: 0.75/2.5$^{n}$', 'L:df1a5525_not_a_neighbour', 'T1'),
]))
E1.append((r'(d) Ageing calibrations of the reference unit (7: 0.25/1, 2025) at the baseline price: $\Delta = \text{value} - I$ (positive: the unit pays), or the difference of the two values (positive: the first calibration gives the higher value)', [
    ('No ageing', 'E:n7_4h_e1_no_ageing:value_minus_I', 'T1'),
    ('Cycling only', 'E:n7_4h_e1_C2:value_minus_I', 'T1'),
    ('Baseline$^{t}$', 'E:n7_4h_e1_C2_calfade:value_minus_I', 'T1'),
    ('Datasheet 8\\,000 cycles', 'E:n7_4h_e1_C4:value_minus_I', 'T1'),
    ('Retention 50\\,\\%', 'E:C3_unit_value_minus_I', 'T1'),
    ('Retention 50\\,\\%, mid-block', 'E:n7_4h_e1_C3_midblock:value_minus_I', 'T1'),
    ('Baseline, 0.50 floor', 'E:soh050:value_minus_I', 'T10'),
    ('No ageing $-$ ret.\\ 50\\,\\%', 'E:n7_4h_e1_no_ageing:vs_C3', 'T1'),
    ('Cycling only $-$ ret.\\ 50\\,\\%', 'E:n7_4h_e1_C2:vs_C3', 'T1'),
    ('Baseline $-$ ret.\\ 50\\,\\%', 'E:n7_4h_e1_C2_calfade:vs_C3', 'T1'),
    ('Datasheet $-$ ret.\\ 50\\,\\%', 'E:n7_4h_e1_C4:vs_C3', 'T1'),
    ('Ret.\\ 50\\,\\%, mid-block $-$ ret.\\ 50\\,\\%', 'E:n7_4h_e1_C3_midblock:vs_C3', 'T1'),
    ('Baseline: 0.50 $-$ 0.70 floor', 'E:soh050:delta_value_vs_070', 'T10'),
]))
used_t1 = {c for _, rows in E1 for _, c, s in rows if s == 'T1'}
dup = {'G:n7_2h_e1_y2030:net', 'G:n7_2h_e1_y2035:net', 'G:n7_4h_e2_y2030:net', 'G:n7_4h_e2_y2035:net', 'CHECK:headline_V_minus_I_settled'}
assert used_t1 | dup == set(T1) and not (used_t1 & dup), set(T1) - used_t1 - dup
# duplicates carry the same numbers as the rows printed
for a, b in [('G:n7_2h_e1_y2030:net', 'G:n7_2h_e1_y2030:gross'), ('G:n7_2h_e1_y2035:net', 'G:n7_2h_e1_y2035:gross'),
             ('G:n7_4h_e2_y2030:net', 'G:n7_4h_e2_y2030:gross'), ('G:n7_4h_e2_y2035:net', 'G:n7_4h_e2_y2035:gross')]:
    for k in ('d gross (k€)', 'threshold / bar (k€)', '× gross', 'd net (k€)', '× net', 'verdict (gross)', 'verdict (net)'):
        assert T1[a][k] == T1[b][k], (a, k)
h, u = T1['CHECK:headline_V_minus_I_settled'], T1['B:n7_4h_e1']
assert h['d gross (k€)'] == '-' + u['d gross (k€)'] and h['× gross'] == u['× gross']
g70, g70b = T10['E:soh050:delta_value_vs_070'], T10['E:soh050:delta_value_vs_g070_REPORT_BESIDE']
assert (g70['d gross (k€)'], g70['×']) == (g70b['d gross (k€)'], g70b['×'])

out = []
P = out.append
P(r"""% =====================================================================================================================
% APPENDIX E -- EVALUATION RECORD. Replaces the submitted Appendix E (\section{Results}, \label{app:results}, and its
% table tab:cs3_results_summary_years_days: per-block results of the submitted five-year, 25-scenario instance).
% The label app:results is kept, so every \ref to it resolves unchanged. All table entries are generated from the W160
% export (frozen_step6_tables_v1_590088fe.json via export_W160/*.csv) by gen_appE.py; none is typed by hand.
% Audited against the frozen JSON by W179a (0 MISMATCH in 1,623 table entries; P5_15_W179A_APPENDIX_E.md). Round 6
% (Addendum 79): row 16 relabelled, E.3/E.4/E.5/E.6 prose corrected; the four W179 check comments closed. Round 7
% (Addendum 80): E.4 second phase and the E.21 note on evaluations 40 and 41.
% =====================================================================================================================
\clearpage
\section{\textcolor{blue}{Evaluation Record}}
\label{app:results}

\textcolor{blue}{
    This appendix gives the record behind the comparisons of Section~\ref{sec:results}: every pairwise comparison (Table~\ref{tab:appE_claims}); every recourse evaluation those comparisons use, with its stopping status, band, gap and slack (Table~\ref{tab:appE_evaluations}); the neighbours of the plan without storage re-evaluated under the certification rule (Table~\ref{tab:appE_neighbours}); the solver starts of the static arrangements (Table~\ref{tab:appE_benchmark}); and the evaluations refused by the gap clause (Table~\ref{tab:appE_gap}). It closes with the certification statistics and the reproducibility record. The multi-scenario evaluations, stopped by the production exit rather than certified, are reported in Table~\ref{tab:res_multiscenario}. All entries are generated from the archived evaluation records. Throughout, $Q$ is the gross operational cost, settlement excluded; the net form subtracts the terminal salvage credit~\eqref{eq:salvage_credit}; the value of a plan is $Q(\boldsymbol{0}) - Q(\boldsymbol{x})$ under the same flexibility price; $F = I + Q$; and $\tau = 4{,}539.07$~\euro{}. Plans are written node: converter rating (MVA) / energy capacity (MWh). Amounts are rounded to 0.1~k\euro{} and ratios to two decimals, except the residual ratio of Table~\ref{tab:appE_gap}, given to three.
}

\subsection{\textcolor{blue}{Comparisons}}
\label{app:results_claims}

\textcolor{blue}{
    Table~\ref{tab:appE_claims} lists the 58 distinct comparisons of the record. Fifty-five are reported both gross and net of the salvage credit; the remaining three, the reference unit at 1.75 times the base flexibility price and the two comparisons at the 0.50 end-of-life floor, were added later and are reported gross only. Each comparison is decided by the rule of~\ref{app:admm_updated_certification}: between two certified evaluations it is determinate if $|\Delta| \geq \max\{3b, 2\tau\}$, with $b$ the larger band; with an uncertified evaluation, if $|\Delta|$ exceeds three times the largest consensus gap and settling slack involved, in gross and gap-corrected terms. Two certificates rest on a turning point at a cycle that was not clean (evaluations~4 and~38); the comparisons that use them are also given in the uncertified form in the note to the table, and none changes verdict.
}
""")
# --- Table E.1 (portrait longtable)
P(r"""{\scriptsize\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{0.265\textwidth} c r r r c r r c@{}}
\caption{\textcolor{blue}{Every pairwise comparison of the record. Eval.: the two evaluations of Table~\ref{tab:appE_evaluations} (reference / other); $^{u}$ uncertified; $^{f}$ flagged certificate. Thr.: the determinacy threshold, or the uncertified bar where an uncertified evaluation is involved; $\times$: $|\Delta|$ divided by it. det.: determinate; w.r.: within resolution; w.b.: within the uncertified bar. Net: net of the terminal salvage credit, against the same threshold or bar. Gross operational cost, settlement excluded.}}
\label{tab:appE_claims}\\
\toprule
 & & & \multicolumn{4}{c}{Gross} & \multicolumn{3}{c}{Net of salvage} \\
\cmidrule(lr){4-7}\cmidrule(l){8-10}
No. & Comparison & Eval. & $\Delta$ (k\euro) & Thr.\ (k\euro) & $\times$ & Verdict & $\Delta$ (k\euro) & $\times$ & Verdict \\
\midrule
\endfirsthead
\multicolumn{10}{@{}l}{\textit{Table~\ref{tab:appE_claims} (continued)}}\\
\toprule
No. & Comparison & Eval. & $\Delta$ (k\euro) & Thr.\ (k\euro) & $\times$ & Verdict & $\Delta$ (k\euro) & $\times$ & Verdict \\
\midrule
\endhead
\bottomrule
\endfoot""")
no = 0
for gi, (title, rows) in enumerate(E1):
    if gi: P(r'\midrule')
    P(r'\multicolumn{10}{@{}p{\dimexpr\linewidth\relax}@{}}{\textit{' + title + r'}}\\[1pt]')
    for desc, claim, src in rows:
        no += 1
        P(e1_row(no, desc, claim, src))
assert no == 58
r3, r26 = T1['B:n7_4h_e3'], T1['J:e4_value_minus_I']
P(r'\midrule')
P(r'\multicolumn{10}{@{}p{\linewidth}@{}}{\textit{Note:} Row~1 is also the headline comparison of Subsection~\ref{subsec:res_baseline}; rows~17--20 are each recorded once gross and once net. In the uncertified form, row~3 has a bar of '
  + r3['uncertified form beside: bar (k€)'] + r'~k\euro{} (' + r3['uncertified form beside: ×'] + r'$\times$) and row~26 a bar of '
  + r26['uncertified form beside: bar (k€)'] + r'~k\euro{} (' + r26['uncertified form beside: ×']
  + r'$\times$, the form used in Table~\ref{tab:res_energy_ladder_m2}), both determinate; rows~30 and~31 already use the uncertified form. $^{\Delta}$: the increase of value $-\,I$ from the smaller to the larger plan (positive: the added capacity pays). Ret.\ 50\,\%: the 50\,\% retention calibration. $^{n}$: not a unit neighbour of the returned plan (its energy capacity at node~7 differs by two lattice steps). $^{t}$: evaluation~56 reproduces evaluation~2 bitwise, so row~48 repeats row~1 with the sign of value $-\,I$. Row~58 against evaluation~61, which reproduces evaluation~2 bitwise through cycle~172, gives the same figures. ---: not evaluated net.}\\')
P(r'\end{longtable}}')

# --- Table E.2 (landscape longtable)
P(r"""
\subsection{\textcolor{blue}{Evaluations}}
\label{app:results_evaluations}

\textcolor{blue}{
    Table~\ref{tab:appE_evaluations} lists the 61 recourse evaluations behind Table~\ref{tab:appE_claims} and the tables of Section~\ref{sec:results}, grouped by the comparisons they serve. For a certified evaluation it gives the certification cycle $k^{*}$, the range of the objective over the certifying window as a fraction of $\tau$ and the band $b$ that enters the determinacy threshold; for an uncertified one, the cycle at which it stopped, the cause, and the priced interface-consensus gap and the settling slack that enter its bar. The last three columns give the number of cycles after the first residual pass with a local solve that was not clean (excluded from certification evidence), the cycle through which a repeated evaluation replayed its earlier record bitwise, and whether the rule was applied during the run or to the committed record of a run made under an earlier version of the rule (Subsection~\ref{subsec:res_certification}).
}

\begin{landscape}
{\scriptsize\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{4.6cm} c l l r r r r r r r r l@{}}
\caption{\textcolor{blue}{Recourse evaluations. Plan: node: MVA/MWh. Condition: flexibility-price multiplier $m$ (base: $m = 1$) or ageing calibration. Status: certified on the oscillatory (osc.) or monotone (mon.) branch, the cause that left the evaluation uncertified (gap clause, lapse reset, growth test), or superseded. $k_0$: first residual pass; $k^{*}$/end: certification cycle, or the cycle at which an uncertified evaluation stopped. Range: window range over $\tau$. Band, and the magnitudes of the gap and slack, in k\euro{}. N.c.: non-clean cycles after $k_0$. Replay: last cycle replayed bitwise. Rule: applied in the run, or to the committed record. $^{f}$: flagged certificate (a turning point at a non-clean cycle); $^{t}$: bitwise twin of evaluation~2; $^{s}$: superseded by evaluation~15; certified under an earlier version of the rule, whose cycle, range and band are shown, and not decided under the rule of \ref{app:admm_updated_certification} within its recorded cycles. ---: not applicable or not tabulated.}}
\label{tab:appE_evaluations}\\
\toprule
No. & Plan & Year & Condition & Status & $k_0$ & $k^{*}$/end & Range/$\tau$ & Band & Gap & Slack & N.c. & Replay & Rule \\
\midrule
\endfirsthead
\multicolumn{14}{@{}l}{\textit{Table~\ref{tab:appE_evaluations} (continued)}}\\
\toprule
No. & Plan & Year & Condition & Status & $k_0$ & $k^{*}$/end & Range/$\tau$ & Band & Gap & Slack & N.c. & Replay & Rule \\
\midrule
\endhead
\bottomrule
\endfoot""")
for gi, (title, rows) in enumerate(G):
    if gi: P(r'\midrule')
    P(r'\multicolumn{14}{@{}l}{\textit{' + title + r'}}\\[1pt]')
    for c, plan, year, cond in rows:
        P(e2_row(c, plan, year, cond))
P(r'\midrule')
P(r'\multicolumn{14}{@{}p{\linewidth}@{}}{\textit{Note:} Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are re-decided from the records of earlier runs; their first residual passes are not tabulated (for evaluation~1, $k_0 = 123$, Fig.~\ref{fig:res_q_cycle}). Evaluation~16 is the first certificate of the 2025 node-5 neighbour; it rested on a cycle with a local solve that did not end optimal, and the neighbour was re-run (evaluation~15).}\\')
P(r'\end{longtable}}')
P(r'\end{landscape}')

# --- Table E.3 neighbours of x = 0
P(r"""
\subsection{\textcolor{blue}{Neighbours of the Plan without Storage}}
\label{app:results_neighbours}

\textcolor{blue}{
    The planning search evaluated all fourteen admissible unit neighbours of the plan without storage at its own resolution ($\sigma_Q$, or the sum of the two evaluations' bars where larger; Subsection~\ref{subsec:res_baseline}). Table~\ref{tab:appE_neighbours} gives the eight of them that were re-evaluated under the certification rule: the smallest unit, 0.25~MVA / 0.5~MWh, at each node installed in 2025 or 2030, and at two pairs of nodes. Every one costs determinately more than the plan without storage. The other six, the smallest unit at the two remaining pairs of nodes and at all three nodes, in 2025 or 2030, were evaluated by the search only, on an earlier state of the code and stopped by the production exit; each cost more than the plan without storage by 3.47--7.17 times the search resolution.
}

\begin{table}[htbp!]
    \centering
    \caption{\textcolor{blue}{Neighbours of the plan without storage re-evaluated under the certification rule: the smallest unit (0.25~MVA / 0.5~MWh) at the nodes listed. $\Delta = F(\boldsymbol{x}) - F(\boldsymbol{0})$, gross operational cost, settlement excluded; every difference is determinate.}}
    \label{tab:appE_neighbours}
    \scriptsize\setlength{\tabcolsep}{5pt}
    \begin{tabular}{l c c r r r}
        \toprule
        Node(s) & Year & Eval. & $\Delta$ (k\euro) & Threshold (k\euro) & $\times$ \\
        \midrule""")
NB = [('5', '2025', 'pb_y2025_n5_v6'), ('7', '2025', 'pb_y2025_n7'), ('9', '2025', 'pb_y2025_n9'),
      ('5', '2030', 'pb_y2030_n5'), ('7', '2030', 'pb_y2030_n7'), ('9', '2030', 'pb_y2030_n9')]
for nd, yr, c in NB:
    r = T5[c]; assert r['v6 verdict'] == 'determinate'
    P(f"        {nd} & {yr} & {ID[c]} & {r['M gross (k€)']} & {r['v6 threshold (k€)']} & {r['×']} \\\\")
for nd, yr, cl, c in [('5 and 9', '2025', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', 'c_156ce2d1'),
                      ('5 and 7', '2030', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'c_6597a79d')]:
    r = T1[cl]; assert r['verdict (gross)'] == 'determinate' and r['other cell'] == c
    P(f"        {nd} & {yr} & {ID[c]} & {r['d gross (k€)']} & {r['threshold / bar (k€)']} & {r['× gross']} \\\\")
assert T1['C:y2025__n5_p0.25_e0.5']['d gross (k€)'] == T5['pb_y2025_n5_v6']['M gross (k€)']
sup = T5['pb_y2025_n5']
P(r"""        \bottomrule
    \end{tabular}
    \par\smallskip
    \begin{minipage}{0.80\linewidth}
        \scriptsize\raggedright
        \textit{Note:} The superseded first certificate of the 2025 node-5 neighbour (evaluation~16) gave """ + sup['M gross (k€)'] + r"""~k\euro{} (""" + sup['×'] + r"""$\times$).
    \end{minipage}
\end{table}""")

# --- Table E.4 benchmark starts
P(r"""
\subsection{\textcolor{blue}{Static Arrangements: Solver Starts}}
\label{app:results_benchmark}

\textcolor{blue}{
    Each static no-reverse-flow arrangement of Subsection~\ref{subsec:res_coordination} was solved from three starts (Table~\ref{tab:appE_benchmark}); the best is reported in Table~\ref{tab:res_benchmark}, and the spread over the starts is its band. Each arrangement is solved in three phases (Section~\ref{sec:case_settings}): the distribution networks' dispatch and the transmission network's dispatch against it, from which the curtailment columns are taken; a re-evaluation of each distribution network at the transmission network's interface voltage, with its decisions held at the first phase and its voltage, thermal and no-reverse-flow rows switched off, in which the exceedance columns count the network-hours (of 864: three networks, twelve blocks, 24~h) whose interface flow exceeds the no-reverse-flow limit and give the largest exceedance; and the consistency pass, a re-solve of each distribution network at that voltage with the rows on, followed by the transmission network's dispatch, whose cost is the $Q$ of Table~\ref{tab:appE_benchmark}. The consistency column is the change of $Q$ from the first phase to the pass.
}

\begin{table}[htbp!]
    \centering
    \caption{\textcolor{blue}{Static no-reverse-flow (NRF) arrangements at the plan without storage, by solver start. Cold: default start; perturbed: perturbed start; warm: started from the certified coordinated evaluation. $Q$: gross operational cost, settlement excluded, after the consistency pass. Curtailment: day-weighted, over the horizon, from the first phase (before the pass).}}
    \label{tab:appE_benchmark}
    \scriptsize\setlength{\tabcolsep}{4pt}
    \begin{tabular}{l l r r r r r r}
        \toprule
        Arrangement & Start & $Q$ & \multicolumn{2}{c}{NRF exceedances} & Consistency & \multicolumn{2}{c}{Curtailment (GWh)} \\
        \cmidrule(lr){4-5}\cmidrule(l){7-8}
         & & (k\euro) & $n$ & largest (p.u.) & pass (k\euro) & TN & DN \\
        \midrule""")
coord = [r for r in T6 if r['arrangement'].startswith('coordinated')][0]
P(r"        Coordinated & --- & " + ks(coord['Q gross (k€)']) + r" & --- & --- & --- & --- & --- \\")
P(r"        \midrule")
SN = {'cold': 'cold', 'perturbed': 'perturbed', 'warm_from_certified': 'warm'}
bests = {}
for arr, lab in [('passive NRF', 'Passive NRF'), ('price-taker NRF', 'Price-taker NRF')]:
    rows = [r for r in T6 if r['arrangement'] == arr and not r['start'].startswith('best')]
    best = [r for r in T6 if r['arrangement'] == arr and r['start'].startswith('best')][0]
    bests[arr] = best
    bstart = best['start'].split(': ')[1]
    for i, r in enumerate(rows):
        mark = '$^{b}$' if r['start'] == bstart else ''
        P(f"        {lab if i == 0 else ''} & {SN[r['start']]}{mark} & {ks(r['Q gross (k€)'])} & {r['NRF violations (n)']} & {r['NRF max excess (p.u.)']} & {neg(r['consistency pass effect (k€)'])} & {r['TN curtailment (GWh, day-weighted)']} & {r['DN curtailment (GWh, day-weighted)']} \\\\")
    if arr == 'passive NRF': P(r"        \midrule")
bp, bt = bests['passive NRF'], bests['price-taker NRF']
P(r"""        \bottomrule
    \end{tabular}
    \par\smallskip
    \begin{minipage}{0.95\linewidth}
        \scriptsize\raggedright
        \textit{Note:} $^{b}$: the start reported in Table~\ref{tab:res_benchmark}; spread over the starts """ + bp['band (k€)'] + r"""~k\euro{} (passive) and """ + bt['band (k€)'] + r"""~k\euro{} (price-taker); reproducibility band of the coordinated evaluation """ + coord['band (k€)'] + r"""~k\euro{}. Without any interface rule, from the cold start, the TN cannot accept the DN exchange in """ + bp['sweep: blocks TN cannot accept (of 12)'] + r""" of 12 blocks (""" + bp['sweep: hours'] + r"""~h; """ + bp['sweep: failing blocks'] + r""") under the passive arrangement and in """ + bt['sweep: blocks TN cannot accept (of 12)'] + r""" of 12 (""" + bt['sweep: hours'] + r"""~h; """ + bt['sweep: failing blocks'].replace('; ', ', ') + r""") under the price-taker arrangement.
    \end{minipage}
\end{table}""")

# --- Table E.5 gap-refused
P(r"""
\subsection{\textcolor{blue}{Evaluations Refused by the Gap Clause}}
\label{app:results_gap}

\textcolor{blue}{
    Table~\ref{tab:appE_gap} lists the evaluations refused by the gap clause, with two others for comparison. At twice the base flexibility price the refused evaluations, and evaluation~42, whose rule was reset by two transmission solver recoveries before the gap clause was reached, share one fingerprint: a priced interface-consensus gap of $-8.8$ to $-9.4$~k\euro{} at the cap, split about 55\,\%, 14\,\% and 31\,\% among nodes~5, 7 and~9, with the power-flow primal residual at 0.68--0.70 of its tolerance in the last cycle (for evaluations~40 and~41 the split and the residual were not recorded). The reference unit at 1.5 times the base price is refused with a smaller gap ($-2.5$~k\euro), the same node split and a residual ratio of 0.027. The 1.25~MVA / 5~MWh plan at the base price is uncertified for a different reason, the growth test after repeated transmission solver recoveries, with a small gap of the opposite sign. The mechanism is described in Subsection~\ref{subsec:res_limitations}.
}

\begin{table}[htbp!]
    \centering
    \caption{\textcolor{blue}{Evaluations refused by the gap clause, and two others with the same or a contrasting signature. Gap: priced interface-consensus gap $t_{\text{sum}}$ at the cap; share: its split among the interface nodes; residual: power-flow primal residual over its tolerance, last cycle; ---: not recorded or none.}}
    \label{tab:appE_gap}
    \scriptsize\setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}r >{\raggedright\arraybackslash}p{3.6cm} c l l r r r r r l@{}}
        \toprule
        Eval. & Plan & Year & Cond. & Cause & Gap & \multicolumn{3}{c}{Share (\%)} & Resid. & Lapse \\
        \cmidrule(lr){7-9}
         & & & & & (k\euro) & n.~5 & n.~7 & n.~9 & & resets \\
        \midrule""")
GORD = ['h_f9eae48f', 'j_5f3cccb4', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5', 'ref:5ca4f86c', 'ref:e28de4ac', 'l_0ee93aca', 'd_36686489']
assert set(GORD) == set(T9)
PL = {c: (p, y, cond) for _, rows in G for c, p, y, cond in rows}
for c in GORD:
    r = T9[c]; p, y, cond = PL[c]
    cause = 'growth test' if r['cause'].startswith('growth') else r['cause']
    cnd = 'base' if cond == 'base' else cond
    P(f"        {ID[c]} & {p} & {y} & {cnd} & {cause} & {neg(r['t_sum at cap (k€)'])} & {d(r['share n5 (%)'])} & {d(r['share n7 (%)'])} & {d(r['share n9 (%)'])} & {d(r['pf_primal last'])} & {d(r['lapse resets at']).replace(',', ', ')} \\\\")
P(r"""        \bottomrule
    \end{tabular}
\end{table}""")

# --- E.6 statistics and reproducibility (moved from Section 4.7)
P(r"""
\subsection{\textcolor{blue}{Certification Statistics and Reproducibility}}
\label{app:results_statistics}

\textcolor{blue}{
    \textbf{Certification statistics.} The 42 single-scenario evaluations of the certification campaign are evaluations 3--11, 15, 22, 23, 26--31, 34--39 and 42--59 of Table~\ref{tab:appE_evaluations}. Of these, 32 certified, 24 on the oscillatory branch and 8 on the monotone branch; the 10 uncertified divide into 5 refused by the gap clause, 2 reset by residual lapses and 3 failed by the growth test. The median certification cycle was 174, and certification came a median 65 cycles after the first residual pass (range 21--89); the median ratio of the window range to $\tau$ at certification was 0.86. The four evaluations added later (32, 33, 60 and 61) all certified. Ten of the 46 certificates the tables use stop within 5\,\% of $\tau$ (evaluations 2, 3, 5, 10, 11, 12, 14, 17, 20 and 23; evaluation~36, at 0.94 on the monotone branch, is just below), so the rule bounds the stopping error of each evaluation at about $\tau$, not well inside it. The three evaluations continued past their certificates, 12, 14 and 3, moved by a further 0.004$\tau$, 0.74$\tau$ and 0.78--0.88$\tau$. After the first residual pass, 21 cycles in 10 evaluations had a local solve that was not clean, all in transmission blocks (column N.c.), while the 119 exits at IPOPT's acceptable level within ten times the tight-tail tolerances, all in distribution blocks, were counted as clean.
}

\textcolor{blue}{
    \textbf{Reproducibility.} Replaying a run reproduces its trajectory bitwise: the multi-scenario evaluation of the plan without storage replayed its 72 recorded cycles bitwise; the three single-scenario reference continuations (the plan without storage, the reference unit and a larger plan not tabulated here) replayed bitwise through the stopping cycles of the residual-based runs they continue, cycles 132, 112 and 87; and every evaluation repeated from an earlier record replayed that record bitwise up to its first residual pass before continuing (column Replay). The evaluations ran one at a time on an Apple M2~Max with 32~GiB of memory under macOS~27.0, single-threaded: OMP, MKL, OpenBLAS, vecLib and NumExpr were capped at one thread in every evaluation's recorded environment, and the HSL library is built without OpenMP, so MA97 ran serially; the solver settings and versions are listed in~\ref{app:admm_updated_settings}. Two distribution blocks (node~7 in 2025 Winter, node~5 in 2035 Winter) end some primary solves at IPOPT's acceptable level within ten times the tight-tail tolerances; they are counted as clean under the rule of~\ref{app:admm_updated_certification}. Two offsets are common to every evaluation and cancel in every reported difference: tightening the interior-point tail lowered the certified objective of each of the three single-scenario references by $1.0$--$1.2 \times 10^{-6}$ relative, with the same sign (the complementarity tolerance went to $10^{-6}$ from $10^{-4}$ in the distribution subproblems and from $5 \times 10^{-4}$ in the transmission subproblem); and the $10^{-5}$~p.u. slack on the renewable availability bound lowers the objective by about 7.8~k\euro{} at first order, $1.2 \times 10^{-5}$ of it.
}
""")
open(sys.argv[1] if len(sys.argv) > 1 else 'appE.tex', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('ok', no, n)
