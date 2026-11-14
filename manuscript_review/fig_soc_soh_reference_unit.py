import csv, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.linewidth':0.6,'xtick.major.width':0.5,'ytick.major.width':0.5,'pdf.fonttype':42})
soc=list(csv.DictReader(open('soc_hourly_reference_unit_2025_2035.csv')))
lim={(r['year']):r for r in csv.DictReader(open('soc_limits_per_year_reference_unit.csv'))}
soh=list(csv.DictReader(open('soh_per_block_calibrations.csv')))
days=['Spring','Summer','Autumn','Winter']; years=['2025','2035']
C={'2025':'#2a78d6','2035':'#eb6834'}
fig=plt.figure(figsize=(6.9,4.6))
gs=fig.add_gridspec(2,4,height_ratios=[1,1.15],hspace=0.55,wspace=0.12)
axs=[fig.add_subplot(gs[0,i]) for i in range(4)]
for ax,d in zip(axs,days):
    for y in years:
        rows=[r for r in soc if r['day']==d and r['year']==y]
        h=[int(r['hour_ending']) for r in rows]; s=[float(r['soc_tso_copy_mwh']) for r in rows]
        ax.plot(h,s,color=C[y],lw=1.4,label=y)
        lo=float(lim[y]['soc_lower_10pct_record_mwh']); up=float(lim[y]['soc_upper_90pct_record_mwh'])
        ax.hlines([lo,up],1,24,colors=C[y],lw=0.6,ls=(0,(2,2)))
    ax.set_title(d,fontsize=8,pad=3); ax.set_xlim(1,24); ax.set_ylim(0,0.85); ax.set_xticks([1,6,12,18,24])
    ax.grid(True,lw=0.3,color='#dddddd'); ax.set_axisbelow(True)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
    if ax is not axs[0]: ax.set_yticklabels([])
axs[0].set_ylabel('Stored energy (MWh)'); fig.text(0.5,0.535,'Hour of the representative day',ha='center',fontsize=8)
axs[0].text(0.02,0.97,'(a)',transform=axs[0].transAxes,va='top',fontsize=9,fontweight='bold')
fig.legend(handles=[Line2D([],[],color=C['2025'],lw=1.4,label='2025'),Line2D([],[],color=C['2035'],lw=1.4,label='2035'),Line2D([],[],color='#555555',lw=0.6,ls=(0,(2,2)),label='10 % and 90 % of available energy')],frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(0.5,0.49),ncol=3,handlelength=2.2)
# panel b
ax=fig.add_subplot(gs[1,:])
cal_order=['Baseline (cycling + calendar)','Cycling only','Retention 50 %','Retention 50 %, mid-block','Datasheet 8 000 cycles','Baseline, 0.50 floor','No ageing']
style={'Baseline (cycling + calendar)':('#2a78d6','-'),'Cycling only':('#eb6834','-'),'Retention 50 %':('#1baf7a','-'),'Retention 50 %, mid-block':('#1baf7a',(0,(4,2))),'Datasheet 8 000 cycles':('#eda100','-'),'Baseline, 0.50 floor':('#2a78d6',(0,(4,2))),'No ageing':('#777777','-')}
names=sorted(set(r['calibration_table6'] for r in soh)); print(names)
xs=[2025,2030,2035,2040]
labels_done=[]
for name in cal_order:
    rows=sorted([r for r in soh if r['calibration_table6'].replace(' ',' ').replace('8,000','8 000')==name or r['calibration_table6']==name],key=lambda r:int(r['block_year']))
    if not rows and name=='No ageing':
        ys=[1,1,1,1]
    elif not rows: print('missing',name); continue
    else: ys=[1.0]+[float(r['soh_end_of_block']) for r in rows]
    c,ls=style[name]; ax.plot(xs,ys,color=c,ls=ls,lw=1.4,marker='o',ms=3.5,mfc='white',mew=1.0)
    labels_done.append((name,ys[-1]))
# direct labels, collision-resolved
lab={'No ageing':('No ageing',0),'Cycling only':('Cycling only',0),'Datasheet 8 000 cycles':('Datasheet 8 000 cycles',0),
     'Baseline (cycling + calendar)':('Baseline; Retention 50 % (end and mid-block): floor binds in 2035',0),
     'Retention 50 %':None,'Retention 50 %, mid-block':None,'Baseline, 0.50 floor':('Baseline, 0.50 floor',-0.022)}
for name,yv in labels_done:
    if lab[name] is None: continue
    txt,dy=lab[name]; ax.annotate(txt,(xs[-1],yv+dy),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color='#222222')
ax.axhline(0.70,color='#999999',lw=0.8,ls=':'); ax.text(2025.1,0.705,'$SoH^{\\mathrm{Min}}=0.70$',fontsize=7,color='#555555',va='bottom')
ax.axhline(0.50,color='#999999',lw=0.8,ls=':'); ax.text(2025.1,0.505,'0.50 floor of the last calibration',fontsize=7,color='#555555',va='bottom')
ax.set_xticks(xs); ax.set_xticklabels(['installation\n(2025)','end of 2025\nblock','end of 2030\nblock','end of 2035\nblock']); ax.set_xlim(2024.5,2050)
ax.set_ylim(0.45,1.03); ax.set_ylabel('State of health'); ax.grid(True,lw=0.3,color='#dddddd'); ax.set_axisbelow(True)
for s_ in ['top','right']: ax.spines[s_].set_visible(False)
ax.text(0.01,0.97,'(b)',transform=ax.transAxes,va='top',fontsize=9,fontweight='bold')
fig.savefig('fig_soc_soh_reference_unit.pdf',bbox_inches='tight'); fig.savefig('fig_soc_soh_reference_unit.png',dpi=200,bbox_inches='tight')
