import csv,re,statistics as st, importlib.util, sys
"""Re-derive the prose counts of Appendix E and the Section 4.6 salvage sentence from export_W160 (zero-solve)."""
import os, tempfile
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location('g',os.path.join(HERE,'gen_appE.py'))
sys.argv=['x',os.path.join(tempfile.mkdtemp(),'appE_check.tex')]
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
ID,T1,T2,T9,T10=g.ID,g.T1,g.T2,g.T9,g.T10
ok=lambda c,m: print(('PASS ' if c else 'FAIL ')+m)
# E.1 row numbering
rows=[]; 
for _,rs in g.E1: rows+=rs
num={claim:i+1 for i,(_,claim,_) in enumerate(rows)}
ok(num['B:n7_4h_e1']==1,'row 1 = reference unit'); ok(num['B:n7_4h_e3']==3,'row 3 = n7 0.75/3')
ok([num[c] for c in ['G:n7_2h_e1_y2030:gross','G:n7_2h_e1_y2035:gross','G:n7_4h_e2_y2030:gross','G:n7_4h_e2_y2035:gross']]==[17,18,19,20],'rows 17-20 = G')
ok(num['J:e4_value_minus_I']==26,'row 26 = J e4'); ok((num['J:e3_to_e4'],num['J:e4_to_e5'])==(30,31),'rows 30,31')
ok(num['L:y2030__n5_p0.25_e0.5__n7_p1_e3']==34,'row 34 = salvage verdict change')
ok(num['E:n7_4h_e1_C2_calfade:value_minus_I']==48,'row 48 = calfade'); ok(num['E:soh050:delta_value_vs_070']==58,'row 58')
ok(ID['d_4a82a64a']==4 and ID['j_a11d7966']==38,'flagged = 4, 38'); ok(ID['e_c2_calfade']==56 and ID['g070_neutrality']==61,'twins 56, 61')
ok(ID['ref:5ca4f86c']==40 and ID['l_0ee93aca']==42 and ID['pb_y2025_n5_v6']==15 and ID['pb_y2025_n5']==16,'40, 42, 15, 16')
# salvage over 55
t1=[T1[c] for _,c,s in rows if s=='T1']; ok(len(t1)==55,'55 T1 comparisons')
sign=[r['claim'] for r in t1 if (float(r['d gross (k€)'])>0)!=(float(r['d net (k€)'])>0)]
ver=[r['claim'] for r in t1 if r['verdict (gross)']!=r['verdict (net)']]
ok(sign==[],'no sign change'); ok(ver==['L:y2030__n5_p0.25_e0.5__n7_p1_e3'],f'one verdict change {ver}')
r=T1['L:y2030__n5_p0.25_e0.5__n7_p1_e3']; ok((r['× gross'],r['× net'])==('0.95','1.22'),'0.95 / 1.22')
G=[T1[c] for c in ['G:n7_2h_e1_y2030:gross','G:n7_2h_e1_y2035:gross','G:n7_4h_e2_y2030:gross','G:n7_4h_e2_y2035:gross']]
vals=[float(x['d gross (k€)']) for x in G]+[float(x['d net (k€)']) for x in G]
ok((min(vals),max(vals))==(85.1,223.6) and all(x['verdict (gross)']==x['verdict (net)']=='determinate' for x in G),'G 85.1-223.6 all det')
# 42 set
ids=list(range(3,12))+[15,22,23]+list(range(26,32))+list(range(34,40))+list(range(42,60))
inv={v:k for k,v in ID.items()}
s42=[inv[i] for i in ids]
exp=[c for c,r in T2.items() if r['table']=='T2' and r['item'] not in ('B','reference')]
ok(sorted(s42)==sorted(exp) and len(s42)==42,'42 = T2 cells less B and references')
cert=[c for c in s42 if T2[c]['status']=='certified']; unc=[c for c in s42 if T2[c]['status']!='certified']
ok(len(cert)==32 and sum(T2[c]['branch']=='oscillatory' for c in cert)==24 and sum(T2[c]['branch']=='monotone' for c in cert)==8,'32 = 24 osc + 8 mon')
causes=[T2[c]['cause (uncertified)'].split(' ')[0] for c in unc]
ok(len(unc)==10 and causes.count('gap')==5 and causes.count('lapse')==2 and causes.count('growth')==3,'10 = 5 gap, 2 lapse, 3 growth')
ks=[int(T2[c]['k* / end']) for c in cert]+[int(T2[c]['k* / end']) for c in unc]
print('  median k*/end over 42:',st.median(ks),' certified only:',st.median([int(T2[c]['k* / end']) for c in cert]))
dk=[int(T2[c]['k* / end'])-int(T2[c]['k0']) for c in cert]; print('  k*-k0 certified: min',min(dk),'median',st.median(dk),'max',max(dk))
print('  median range/tau certified:',st.median([float(T2[c]['range/τ']) for c in cert]))
later=[inv[i] for i in (32,33,60,61)]; ok(all(T2[c]['table']=='T10' and T2[c]['status']=='certified' for c in later),'32,33,60,61 = T10 certified')
ten=sorted(ID[c] for c,r in T2.items() if r['at_or_above_0.95_tau_counted']=='yes'); ok(ten==[2,3,5,10,11,12,14,17,20,23],f'ten {ten}')
certall=[c for c,r in T2.items() if r['status']=='certified']; ok(len(certall)-3==46,'46 = 49 certified - superseded - 2 twins')
ok(T2['i_5a6a88b4']['range/τ']=='0.94' and ID['i_5a6a88b4']==36,'36 at 0.94 monotone')
nc=[(c,int(T2[c]['non-clean cycles after N'])) for c in s42 if T2[c]['non-clean cycles after N'] not in ('','0')]
ok(sum(n for _,n in nc)==21 and len(nc)==10,f'21 cycles in 10 evaluations {nc}')
ok(ID['b_4649234b']==14 and ID['d_c52e1670']==3,'continued 14, 3')
# E.5
fp=[T9[c] for c in ['j_5f3cccb4','l_45aa25a6','l_7c455554','l_b2251bc5','l_0ee93aca','ref:5ca4f86c','ref:e28de4ac']]
gaps=[float(r['t_sum at cap (k€)']) for r in fp]; ok((min(gaps),max(gaps))==(-9.4,-8.8),'gap -8.8..-9.4')
pf=[float(r['pf_primal last']) for r in fp if r['pf_primal last']]; ok(0.68<=min(pf) and max(pf)<=0.70,f'pf {min(pf)}-{max(pf)}')
sh=[(float(r['share n5 (%)']),float(r['share n7 (%)']),float(r['share n9 (%)'])) for r in fp if r['share n5 (%)']]
ok(all(round(a)==55 and round(b)==14 and round(c)==31 for a,b,c in sh),'55/14/31')
h=T9['h_f9eae48f']; ok((h['t_sum at cap (k€)'],h['pf_primal last'])==('-2.5','0.027'),'unit 1.5: -2.5, 0.027')
ok(T9['d_36686489']['t_sum at cap (k€)']=='1.1','5 MWh +1.1')
# E.3
ok(all(T2[c]['status']=='certified' for c in ['pb_y2025_n5_v6','pb_y2025_n7','pb_y2025_n9','pb_y2030_n5','pb_y2030_n7','pb_y2030_n9','c_156ce2d1','c_6597a79d']),'E.3 eight certified')
