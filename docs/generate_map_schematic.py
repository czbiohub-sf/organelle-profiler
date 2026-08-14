"""
Generate distinctiveness mAP schematic figure.
Output: docs/distinctiveness_map_schematic.png

Usage:
    python docs/generate_map_schematic.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

FW, FH = 26, 15
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor('white')

CB='#2166ac'; CR='#d6604d'; CG='#1a9850'; CY='#e6ab02'
CP='#762a83'; CT='#01665e'; CO='#b35806'; CK='#d73027'
CGRAY='#888888'; CLIGHT='#f4f4f4'; CDARK='#1a1a1a'; CNTC='#aaaaaa'

def add_ax(l,b,w,h): return fig.add_axes([l,b,w,h])
def ft(x,y,s,**kw): return fig.text(x,y,s,**kw)

np.random.seed(42)

GCOLS=[CB,CR,CG,CP,CY,CT,CK,CO]
GCTR=[(-2.1,1.4),(-1.0,-1.7),(0.9,1.9),(2.2,0.5),
      (-2.7,-0.5),(1.6,-1.9),(0.1,0.2),(2.7,2.3)]
CATMAP={0:'Traf',1:'Traf',5:'Traf',2:'Meta',4:'Meta',
        6:'Trans',7:'Trans',3:'Cell'}
CATCOL={'Traf':CB,'Meta':CG,'Trans':CR,'Cell':CY}

PTS=[]
for i,(cx,cy) in enumerate(GCTR):
    for _ in range(5):
        PTS.append((cx+np.random.randn()*0.22, cy+np.random.randn()*0.22, i))
NTC=[(np.random.randn()*0.30, np.random.randn()*0.28) for _ in range(14)]

def scatter_base(ax):
    ax.set_facecolor(CLIGHT)
    ax.set_xlim(-3.8,3.8); ax.set_ylim(-3.2,3.2)
    ax.set_xlabel('Embedding dim 1', fontsize=10, labelpad=2)
    ax.set_ylabel('Embedding dim 2', fontsize=10, labelpad=2)
    ax.tick_params(labelsize=9, length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#cccccc')

# ── Section headers ───────────────────────────────────────────────────────
ft(0.012,0.975,'A',fontsize=19,fontweight='black',color=CDARK)
ft(0.030,0.975,'Three mAP metric types',fontsize=14,fontweight='bold',color=CDARK)
ft(0.012,0.465,'B',fontsize=19,fontweight='black',color=CDARK)
ft(0.030,0.465,'Reporter subspace vs global embedding',fontsize=14,fontweight='bold',color=CDARK)

# ── A1: Activity ─────────────────────────────────────────────────────────
AX,AY,AW,AH = 0.028,0.525,0.128,0.385
ax1 = add_ax(AX,AY,AW,AH)
scatter_base(ax1)
for x,y in NTC:
    ax1.scatter(x,y,c=CNTC,s=45,alpha=0.75,zorder=3,edgecolors='white',linewidths=0.5)
ax1.add_patch(Ellipse((0,0),1.4,1.1,fill=True,facecolor='#e0e0e0',
                       edgecolor=CNTC,lw=1.5,zorder=2,alpha=0.5))
ax1.text(0,-0.04,'NTC',fontsize=10,ha='center',va='center',
         color='#666666',fontweight='bold',zorder=5)
for x,y,i in PTS:
    if i != 0:
        ax1.scatter(x,y,c=GCOLS[i],s=20,alpha=0.12,zorder=2,edgecolors='none')
qpts=[(p[0],p[1]) for p in PTS if p[2]==0]
for x,y in qpts:
    ax1.scatter(x,y,c=CB,s=72,alpha=0.92,zorder=5,edgecolors=CDARK,linewidths=1.5)
qcx=np.mean([p[0] for p in qpts]); qcy=np.mean([p[1] for p in qpts])
ntc_cx=np.mean([p[0] for p in NTC]); ntc_cy=np.mean([p[1] for p in NTC])
for rx,ry in qpts:
    ax1.plot([rx,ntc_cx],[ry,ntc_cy],color=CGRAY,lw=0.8,alpha=0.55,zorder=1,
             linestyle=(0,(4,3)))
ax1.text(qcx,qcy+0.65,'LAMP1 KO\nguides',fontsize=10,color=CB,
         fontweight='bold',ha='center',zorder=6)
# question inside plot at bottom
ax1.text(0,-2.85,'Does this geneKO cause a detectable\nphenotypic change from NTCs?',
         fontsize=9.5,ha='center',va='center',color=CDARK,zorder=7,
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#cccccc',lw=0.8,alpha=0.92))

ft(AX+AW/2,AY+AH+0.014,'Activity mAP',ha='center',fontsize=12,fontweight='bold',color=CDARK)
ft(AX+AW/2,AY+AH+0.003,'pos: same geneKO reps  |  neg: NTC reps',
   ha='center',fontsize=9,color=CGRAY,style='italic')
ft(AX+AW/2,AY+AH-0.009,'dot = guide replicate   color = geneKO',
   ha='center',fontsize=8.5,color='#aaaaaa',style='italic')

# ── A2: Distinctiveness ───────────────────────────────────────────────────
BX,BY,BW,BH = 0.175,0.525,0.128,0.385
ax2 = add_ax(BX,BY,BW,BH)
scatter_base(ax2)
for sp in ax2.spines.values():
    sp.set_linewidth(0.8); sp.set_color('#cccccc')
ax2.set_facecolor(CLIGHT)
for x,y,i in PTS:
    ax2.scatter(x,y,c=GCOLS[i],s=52,alpha=0.80,zorder=3,edgecolors='white',linewidths=0.3)
qpts=[(p[0],p[1]) for p in PTS if p[2]==0]
for x,y in qpts:
    ax2.scatter(x,y,c=CB,s=70,zorder=5,edgecolors=CDARK,linewidths=1.8)
qcx2=np.mean([p[0] for p in qpts]); qcy2=np.mean([p[1] for p in qpts])
ax2.text(qcx2,qcy2+0.65,'LAMP1 KO\nguides',fontsize=10,color=CB,
         fontweight='bold',ha='center',zorder=6)
for idx in [1,2,3,4,5,6,7]:
    ax2.annotate('',xy=(GCTR[idx][0],GCTR[idx][1]),
                 xytext=(qcx2,qcy2),
                 arrowprops=dict(arrowstyle='->',color='#aaaaaa',lw=0.85,
                                 connectionstyle=f'arc3,rad={0.07+idx*0.07}'))
# question inside plot at bottom
ax2.text(0,-2.85,'Is this geneKO phenotype unique\namong ALL other geneKOs?',
         fontsize=9.5,ha='center',va='center',color=CDARK,zorder=7,
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#cccccc',lw=0.8,alpha=0.92))

ft(BX+BW/2,BY+BH+0.014,'Distinctiveness mAP',ha='center',fontsize=12,fontweight='bold',color=CDARK)
ft(BX+BW/2,BY+BH+0.003,'pos: same geneKO reps  |  neg: ALL other geneKOs',
   ha='center',fontsize=9,color=CGRAY,style='italic')
ft(BX+BW/2,BY+BH-0.009,'dot = guide replicate   color = geneKO',
   ha='center',fontsize=8.5,color='#aaaaaa',style='italic')

# ── A3: Consistency ───────────────────────────────────────────────────────
CX2,CY2,CW,CH = 0.322,0.525,0.128,0.385
ax3 = add_ax(CX2,CY2,CW,CH)
scatter_base(ax3)
pathway_genes = {
    'Pathway A\n(Trafficking)': {'col':CB,'pts':[(-2.1,1.4),(-1.8,1.1),(-2.3,1.0),(-1.6,1.7)]},
    'Pathway B\n(Metabolism)':  {'col':CG,'pts':[(1.6,1.8),(1.9,1.4),(1.4,1.5),(2.0,1.1)]},
    'Pathway C\n(Translation)': {'col':CR,'pts':[(-1.1,-1.6),(-0.8,-1.9),(-1.4,-1.2),(-0.9,-1.4)]},
}
np.random.seed(7)
ctrs={}
for pname,pdata in pathway_genes.items():
    col=pdata['col']
    spts=[]
    for cx,cy in pdata['pts']:
        x=cx+np.random.randn()*0.14; y=cy+np.random.randn()*0.14
        spts.append((x,y))
        ax3.scatter(x,y,c=col,s=65,alpha=0.88,zorder=4,edgecolors=CDARK,linewidths=1.2)
    for ii in range(len(spts)):
        for jj in range(ii+1,len(spts)):
            ax3.plot([spts[ii][0],spts[jj][0]],[spts[ii][1],spts[jj][1]],
                     color=col,lw=1.0,alpha=0.45,zorder=3)
    xs=[p[0] for p in spts]; ys=[p[1] for p in spts]
    ax3.add_patch(Ellipse((np.mean(xs),np.mean(ys)),1.3,0.85,angle=10,
                           fill=False,edgecolor=col,lw=2.0,linestyle='--',alpha=0.8,zorder=5))
    ctrs[pname]=(np.mean(xs),np.mean(ys))
    offsets={'Pathway A\n(Trafficking)':(0.0,0.65),
             'Pathway B\n(Metabolism)':(0.6,0.0),
             'Pathway C\n(Translation)':(0.0,-0.65)}
    dx,dy=offsets.get(pname,(0,0.5))
    ax3.text(np.mean(xs)+dx,np.mean(ys)+dy,pname,fontsize=9,color=col,
             fontweight='bold',ha='center',va='center',zorder=6)
for x,y,i in PTS:
    if CATMAP.get(i,'Other') not in ('Traf','Meta','Trans'):
        ax3.scatter(x,y,c='#cccccc',s=18,alpha=0.25,zorder=2,edgecolors='none')
pnames=list(ctrs.keys())
for ia,ib in [(0,1),(1,2)]:
    pa=ctrs[pnames[ia]]; pb=ctrs[pnames[ib]]
    mx=(pa[0]+pb[0])/2; my=(pa[1]+pb[1])/2
    ax3.annotate('',xy=pb,xytext=pa,
                 arrowprops=dict(arrowstyle='<->',color='#cccccc',lw=1.2,
                                 connectionstyle='arc3,rad=0.15'))
    ax3.text(mx,my,'neg\npairs',fontsize=8,ha='center',va='center',
             color='#999999',style='italic',
             bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.8))
# question inside plot at bottom
ax3.text(0,-2.85,'Do pathway members separate\nfrom all other geneKOs?',
         fontsize=9.5,ha='center',va='center',color=CDARK,zorder=7,
         bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#cccccc',lw=0.8,alpha=0.92))

ft(CX2+CW/2,CY2+CH+0.014,'Consistency mAP',ha='center',fontsize=12,fontweight='bold',color=CDARK)
ft(CX2+CW/2,CY2+CH+0.003,'pos: same pathway  |  neg: ALL other geneKOs',
   ha='center',fontsize=9,color=CGRAY,style='italic')
ft(CX2+CW/2,CY2+CH-0.009,'dot = geneKO   color = pathway',
   ha='center',fontsize=8.5,color='#aaaaaa',style='italic')

ft(0.232,0.488,
   '◀── all three: normalized by global baseline → reporter[cat] / all_reporters[cat] ──▶',
   ha='center',fontsize=9.5,color=CGRAY,style='italic')

# ── Panel B ───────────────────────────────────────────────────────────────
cat_info={
    'Mem.Traffic':(CB,[(-1.8,1.2),(-1.5,0.7),(-2.1,0.5)]),
    'Metabolism': (CG,[(1.6,1.8),(1.9,1.3),(1.4,1.5)]),
    'Translation':(CR,[(-1.2,-1.5),(-0.8,-1.8),(-1.5,-1.1)]),
    'Signaling':  (CP,[(1.2,-1.2),(1.6,-0.8),(0.9,-1.5)]),
    'Cell Cycle': (CY,[(0.1,0.2),(0.5,-0.2),(-0.2,0.4)]),
}
def draw_embed(ax,al,ab,aw,ah,spread_map,highlight,title,sub1,sub2):
    """spread_map: dict cat->spread, or single float for uniform spread."""
    np.random.seed(5)
    ax.set_facecolor(CLIGHT)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    ax.set_xlabel('dim 1',fontsize=10,labelpad=2)
    ax.set_ylabel('dim 2',fontsize=10,labelpad=2)
    ax.tick_params(labelsize=9,length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('#cccccc')
    for cat,(col,ctrs2) in cat_info.items():
        spread = spread_map[cat] if isinstance(spread_map,dict) else spread_map
        hl = cat in highlight
        for cx,cy in ctrs2:
            for _ in range(5):
                x=cx+np.random.randn()*spread; y=cy+np.random.randn()*spread
                ax.scatter(x,y,c=col,s=52 if hl else 38,
                           alpha=0.88 if hl else 0.55,zorder=3,edgecolors='none')
        if hl:
            ax.add_patch(Ellipse(
                (np.mean([c[0] for c in ctrs2]),np.mean([c[1] for c in ctrs2])),
                1.3,0.9,angle=15,fill=False,edgecolor=col,lw=2,linestyle='--',zorder=4))
    ft(al+aw/2,ab+ah+0.018,title,ha='center',fontsize=12,fontweight='bold',color=CDARK)
    ft(al+aw/2,ab+ah+0.007,sub1,ha='center',fontsize=9.5,color=CDARK)
    ft(al+aw/2,ab+ah-0.005,sub2,ha='center',fontsize=9,color=CGRAY,style='italic')

B1L,B1B,B1W,B1H=0.028,0.065,0.188,0.360
B2L,B2B,B2W,B2H=0.240,0.065,0.188,0.360
ax_b1=add_ax(B1L,B1B,B1W,B1H)
# Global: all categories well-separated (tight spread for all)
draw_embed(ax_b1,B1L,B1B,B1W,B1H,0.22,[],'Global embedding',
           'All ~1500 PCs, all reporters combined','→ all categories cleanly separated')
ax_b2=add_ax(B2L,B2B,B2W,B2H)
# LAMP1 subspace: only Mem.Traffic tight, others mixed (large spread)
lamp1_spread={'Mem.Traffic':0.19,'Metabolism':0.85,'Translation':0.85,
              'Signaling':0.85,'Cell Cycle':0.85}
draw_embed(ax_b2,B2L,B2B,B2W,B2H,lamp1_spread,['Mem.Traffic'],'LAMP1 subspace',
           '31 PCs — lysosome/trafficking lens','→ only trafficking genes sharply separated')
leg_b=[mpatches.Patch(color=v[0],label=k) for k,v in cat_info.items()]
ax_b2.legend(handles=leg_b,fontsize=8,loc='lower right',framealpha=0.9,
             title='category',title_fontsize=8.5,handlelength=1,borderpad=0.4,labelspacing=0.3)
fig.text(0.228,0.245,'→',fontsize=30,color=CGRAY,ha='center',va='center')

# ── Panel C: Flowchart ────────────────────────────────────────────────────
ft(0.458,0.975,'C',fontsize=19,fontweight='black',color=CDARK)
ft(0.474,0.975,'Normalized distinctiveness mean_mAP — computation pipeline',
   fontsize=14,fontweight='bold',color=CDARK)
CG_LIGHT='#a8a8a8'; CB_LIGHT='#7bacd4'; CP_LIGHT='#a87aba'; CK_LIGHT='#555555'

ax_c=add_ax(0.455,0.12,0.540,0.820)
ax_c.set_xlim(0,10); ax_c.set_ylim(0,9.5)
ax_c.axis('off')

def cbox(x,y,w,h,lines,fc,fontsize=9,tc='white'):
    ax_c.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.10',facecolor=fc,edgecolor='white',lw=1.0,zorder=3))
    if isinstance(lines,str): lines=[lines]
    dy=h/(len(lines)+1)
    for j,ln in enumerate(lines):
        ax_c.text(x,y+h/2-dy*(j+1),ln,fontsize=fontsize,ha='center',va='center',
                  color=tc,fontweight='bold',zorder=4)
def carr(x1,y1,x2,y2,col='#aaaaaa'):
    ax_c.annotate('',xy=(x2,y2),xytext=(x1,y1),
                  arrowprops=dict(arrowstyle='-|>',color=col,lw=1.2,mutation_scale=12))

cbox(5,9.1,9.0,0.52,
     ['PCA-optimized guide embedding',
      'obs = geneKO replicates   ×   vars = reporter PCA components'],
     CK_LIGHT,fontsize=9)
carr(2.5,8.84,2.5,8.42); carr(7.5,8.84,7.5,8.42)
cbox(2.5,8.18,4.2,0.42,['All reporters (~1500 PCs)  →  global baseline'],CG_LIGHT)
cbox(7.5,8.18,4.2,0.42,['Single reporter (30–50 PCs)  →  per-reporter ×37'],CB_LIGHT)
carr(2.5,7.97,2.5,7.50); carr(7.5,7.97,7.5,7.50)
cbox(2.5,7.24,4.2,0.46,
     ['Distinctiveness mAP','pos: same KO guides | neg: all others'],CG_LIGHT)
cbox(7.5,7.24,4.2,0.46,
     ['Distinctiveness mAP','pos: same KO guides | neg: all others'],CB_LIGHT)
ax_c.text(0.55,7.24,'same\nformula',fontsize=8,ha='center',va='center',
          color='#bbbbbb',style='italic')
carr(2.5,7.01,2.5,6.55); carr(7.5,7.01,7.5,6.55)
cbox(2.5,6.29,4.2,0.42,['Per-category mean(mAP)  →  baseline[cat]'],CG_LIGHT)
cbox(7.5,6.29,4.2,0.42,['Per-category mean(mAP)  →  reporter[cat]'],CB_LIGHT)
carr(2.5,6.08,3.8,5.62); carr(7.5,6.08,6.2,5.62)
cbox(5.0,5.28,5.2,0.62,
     ['Normalization:  spoke[cat]  =  reporter[cat]  /  baseline[cat]',
      'spoke = signal captured by reporter / signal captured by all reporters'],
     CP_LIGHT,fontsize=9)
cats=['Mem.\nTraffic','Metab.','Cell\nCycle','Prot.\nHomeo.',
      'Signal.','Cytosk.','Gene\nExpr.','Transl.']
n=len(cats)
ang=np.linspace(0,2*np.pi,n,endpoint=False).tolist()+[0]
# scale each reporter so its max spoke = 1.0 (outer ring)
raw_L=np.array([2.1,0.4,0.5,0.9,0.6,0.7,0.7,0.5])
raw_T=np.array([0.5,2.3,0.4,0.8,0.6,0.7,0.7,0.6])
vL=(raw_L/raw_L.max()).tolist()+[raw_L[0]/raw_L.max()]
vT=(raw_T/raw_T.max()).tolist()+[raw_T[0]/raw_T.max()]
ax_r=fig.add_axes([0.578,0.210,0.245,0.285],polar=True)
ax_r.set_facecolor('#f9f9f9')
ax_r.fill(ang,vL,alpha=0.18,color=CB,zorder=3)
ax_r.plot(ang,vL,'o-',color=CB,lw=2.0,ms=4.5,zorder=4,label='LAMP1')
ax_r.fill(ang,vT,alpha=0.18,color=CG,zorder=3)
ax_r.plot(ang,vT,'o-',color=CG,lw=2.0,ms=4.5,zorder=4,label='TOMM20')
ax_r.set_xticks(ang[:-1]); ax_r.set_xticklabels(cats,fontsize=9)
ax_r.set_ylim(0,1.0)
ax_r.set_yticks([0.5,1.0]); ax_r.set_yticklabels(['0.5','1.0'],fontsize=8,color='#888888')
ax_r.tick_params(pad=5)
ax_r.legend(loc='upper right',bbox_to_anchor=(1.52,1.12),fontsize=9.5,
            framealpha=0.9,handlelength=1.5)
ft(0.700,0.516,'Example output — normalized mean_mAP\n(reporter / all-reporters baseline)',
   ha='center',fontsize=10,fontweight='bold',color=CDARK)
ax_c.text(3.2,0.85,'LAMP1: trafficking\nhighest enrichment',fontsize=9,ha='center',va='center',
    color=CB,bbox=dict(boxstyle='round,pad=0.25',fc='#e8f0f8',ec=CB,lw=0.8))
ax_c.text(6.8,0.85,'TOMM20: metabolism\nhighest enrichment',fontsize=9,ha='center',va='center',
    color=CG,bbox=dict(boxstyle='round,pad=0.25',fc='#e8f5e9',ec=CG,lw=0.8))
ax_c.add_patch(FancyBboxPatch((0.1,0.10),9.8,0.36,
    boxstyle='round,pad=0.08',facecolor='#555555',edgecolor='none',zorder=3))
ax_c.text(5.0,0.28,
    'all cells (650k–40M per reporter)   |   downsampled (~750k cells)',
    fontsize=9,ha='center',va='center',color='white',fontweight='bold',zorder=4)

import os
out = os.path.join(os.path.dirname(__file__), 'distinctiveness_map_schematic.png')
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')

if __name__ == '__main__':
    pass
