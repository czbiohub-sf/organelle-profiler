"""
Panel-A-only variant of `generate_map_schematic.py` — emits ONLY the
three mAP-metric sub-panels (Activity / Distinctiveness / Consistency)
in a slide-friendly large-text layout.

Layout (top → bottom inside each sub-panel column):
    sub-panel title (mAP name)         — fontsize 56
    pos/neg subtitle                   — fontsize 40
    dot/color legend subtitle          — fontsize 36
    scatter axes                        — embedding dim plot
    x-axis label                        — auto
    "question" caption                  — fontsize 38, BELOW x-axis
    bottom global-baseline note         — fontsize 44, fig-wide

Output: docs/distinctiveness_map_schematic_panelA.png

Usage:
    python docs/generate_map_schematic_panelA.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

FW, FH = 30, 19
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor('white')

CB='#2166ac'; CR='#d6604d'; CG='#1a9850'; CY='#e6ab02'
CP='#762a83'; CT='#01665e'; CO='#b35806'; CK='#d73027'
CGRAY='#888888'; CLIGHT='#f4f4f4'; CDARK='#1a1a1a'; CNTC='#aaaaaa'
# Darker frame/tick palette so the axes hold up against the now-much-
# larger dots + text. The original `#cccccc` washed out at 4× font.
CFRAME='#444444'
CTICKLBL='#222222'

def add_ax(l, b, w, h):
    return fig.add_axes([l, b, w, h])

def ft(x, y, s, **kw):
    return fig.text(x, y, s, **kw)

np.random.seed(42)

GCOLS = [CB, CR, CG, CP, CY, CT, CK, CO]
GCTR = [(-2.1, 1.4), (-1.0, -1.7), (0.9, 1.9), (2.2, 0.5),
        (-2.7, -0.5), (1.6, -1.9), (0.1, 0.2), (2.7, 2.3)]
CATMAP = {0: 'Traf', 1: 'Traf', 5: 'Traf', 2: 'Meta', 4: 'Meta',
          6: 'Trans', 7: 'Trans', 3: 'Cell'}

PTS = []
for i, (cx, cy) in enumerate(GCTR):
    # 4 guide replicates per geneKO (typical OPS guide library: 4
    # sgRNAs per gene). Used by both Plot A (Activity) and Plot B
    # (Distinctiveness) so the on-screen replicate count matches.
    for _ in range(4):
        PTS.append((cx + np.random.randn() * 0.22,
                    cy + np.random.randn() * 0.22, i))
NTC = [(np.random.randn() * 0.30, np.random.randn() * 0.28) for _ in range(14)]


def scatter_base(ax, show_ylabel=False):
    """Set up a sub-panel scatter axes.  `show_ylabel` is True only for
    the leftmost panel — the other two share the y-scale visually with
    panel A and don't need to repeat the axis title."""
    ax.set_facecolor(CLIGHT)
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel('Embedding dim 1', fontsize=40, labelpad=8,
                  color=CTICKLBL)
    if show_ylabel:
        ax.set_ylabel('Embedding dim 2', fontsize=40, labelpad=8,
                      color=CTICKLBL)
    # Bigger ticks + darker color to match the heavyweight text.
    ax.tick_params(labelsize=32, length=10, width=2.0,
                    colors=CTICKLBL)
    # Plots B/C hide their y-tick numerals — same scale as plot A,
    # readers don't need the redundant labels. Ticks themselves stay so
    # the grid still reads correctly.
    if not show_ylabel:
        ax.tick_params(axis='y', labelleft=False)
    for sp in ax.spines.values():
        sp.set_linewidth(2.2); sp.set_color(CFRAME)
    # Light grid for readability — the eye needs *some* graticule to
    # judge the embedding-space distance between guide clusters and
    # the NTC blob.
    ax.grid(True, color=CFRAME, alpha=0.30, linewidth=0.8, zorder=1)


# Sub-panel geometry. Header (mAP name + 2 subtitles) sits in
# [0.83, 0.96] above each axes; the axes themselves span [0.32, 0.80]
# vertically. Below the axes: x-axis label auto-positions ~y=0.27;
# question caption at y=0.15; global baseline note at y=0.04.
SUB_Y, SUB_H = 0.32, 0.48
# Layout: 3.75% margin | 27.5% panel | 5% gap | 27.5% panel | 5% gap |
# 27.5% panel | 3.75% margin  =  100% width. Sub-panels are slightly
# wider and the gap is tighter than the previous (25% / 7.5%) layout —
# titles still don't bleed because they're shorter than the panel.
SUB_W = 0.290
GAP = 0.025
PANEL_LEFTS = [0.0375,
               0.0375 + SUB_W + GAP,
               0.0375 + 2 * (SUB_W + GAP)]

# Header y-coords (fixed per row). With 4× fontsizes the legend-style
# third subtitle ("dot = guide replicate") just added visual noise and
# overlapped subtitle 1 — dropped entirely. Subtitle 1 is now a
# multi-line "pos: …\nneg: …" so each line fits inside the sub-panel
# width instead of spilling into the neighbor.
TITLE_Y    = 0.90      # mAP name — gap to subtitle reduced 0.07 → 0.06
SUBTITLE_Y = 0.84      # subtitle gap to axes top reduced 0.05 → 0.04
# QUESTION_Y is the CENTER of a 2-line, 36pt question block (box
# half-height ~0.046 with the round pad=0.5 bbox at FH=19). x-axis
# label bottom is at fig-y ~0.260. At 0.19 the box top ~0.236 leaves
# a ~0.024 gap below the xlabel — pulled up from 0.17 to bring the
# question closer to the plot like the user asked.
QUESTION_Y = 0.19

# ── A1: Activity ──────────────────────────────────────────────────────────
AX = PANEL_LEFTS[0]
ax1 = add_ax(AX, SUB_Y, SUB_W, SUB_H)
scatter_base(ax1, show_ylabel=True)
for x, y in NTC:
    ax1.scatter(x, y, c=CNTC, s=260, alpha=0.88, zorder=3,
                edgecolors=CDARK, linewidths=2.2)
# Dashed outline sized from the actual NTC scatter so the ring
# always encompasses every dot regardless of seed-to-seed jitter.
# Margin of 0.5 in each axis adds breathing room around the outliers.
_ntc_xs = [p[0] for p in NTC]; _ntc_ys = [p[1] for p in NTC]
_ntc_cx_e = (max(_ntc_xs) + min(_ntc_xs)) / 2
_ntc_cy_e = (max(_ntc_ys) + min(_ntc_ys)) / 2
_ntc_w   = (max(_ntc_xs) - min(_ntc_xs)) + 0.7
_ntc_h   = (max(_ntc_ys) - min(_ntc_ys)) + 0.7
ax1.add_patch(Ellipse((_ntc_cx_e, _ntc_cy_e), _ntc_w, _ntc_h, fill=False,
                       edgecolor=CNTC, lw=3.2, linestyle='--',
                       alpha=0.85, zorder=2))
# Label placed just off the NTC ring (bottom-right of the ellipse)
# so it doesn't sit on top of the dots / ellipse.
ax1.text(_ntc_cx_e + _ntc_w / 2 - 0.05, _ntc_cy_e - _ntc_h / 2 + 0.35,
         'NTCs\n(controls)', fontsize=36, ha='left',
         va='top', color='#444444', fontweight='bold', zorder=5,
         linespacing=1.15)
for x, y, i in PTS:
    if i != 0:
        ax1.scatter(x, y, c=GCOLS[i], s=80, alpha=0.45, zorder=2,
                    edgecolors='none')
qpts = [(p[0], p[1]) for p in PTS if p[2] == 0]
for x, y in qpts:
    ax1.scatter(x, y, c=CB, s=260, alpha=0.95, zorder=5,
                edgecolors=CDARK, linewidths=2.2)
qcx = np.mean([p[0] for p in qpts]); qcy = np.mean([p[1] for p in qpts])
ntc_cx = np.mean([p[0] for p in NTC]); ntc_cy = np.mean([p[1] for p in NTC])
for rx, ry in qpts:
    ax1.plot([rx, ntc_cx], [ry, ntc_cy], color=CGRAY, lw=1.4, alpha=0.6,
             zorder=1)
ax1.text(qcx, qcy + 0.45, 'LAMP1 KO\nguides', fontsize=36, color=CB,
         fontweight='bold', ha='center', zorder=6)
# Label the dashed LAMP1→NTC connectors as the negative-pair lines.
# Anchor on the midpoint between LAMP1 centroid and NTC centroid. No
# bbox here — a white-fill box would hide the dashed neg-pair lines
# we're trying to label.
ax1.text((qcx + ntc_cx) / 2, (qcy + ntc_cy) / 2,
         'neg\npairs', fontsize=32, ha='center', va='center',
         color='#888888', style='italic', zorder=7)

ft(AX + SUB_W / 2, TITLE_Y, 'Activity mAP',
   ha='center', fontsize=56, fontweight='bold', color=CDARK)
ft(AX + SUB_W / 2, SUBTITLE_Y,
   'pos: same geneKO reps\nneg: NTC reps',
   ha='center', va='center', fontsize=34, color=CGRAY, style='italic',
   linespacing=1.2)
ft(AX + SUB_W / 2, QUESTION_Y,
   'Is this geneKO\nactive vs NTCs?',
   ha='center', va='center', fontsize=36, color=CDARK,
   linespacing=1.25,
   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=CFRAME,
             lw=1.5, alpha=0.95))

# ── A2: Distinctiveness ───────────────────────────────────────────────────
BX = PANEL_LEFTS[1]
ax2 = add_ax(BX, SUB_Y, SUB_W, SUB_H)
scatter_base(ax2)
for x, y, i in PTS:
    ax2.scatter(x, y, c=GCOLS[i], s=260, alpha=0.92, zorder=3,
                edgecolors=CDARK, linewidths=2.2)
qpts = [(p[0], p[1]) for p in PTS if p[2] == 0]
for x, y in qpts:
    ax2.scatter(x, y, c=CB, s=260, zorder=5,
                edgecolors=CDARK, linewidths=2.2)
qcx2 = np.mean([p[0] for p in qpts]); qcy2 = np.mean([p[1] for p in qpts])
ax2.text(qcx2, qcy2 + 0.45, 'LAMP1 KO\nguides', fontsize=36, color=CB,
         fontweight='bold', ha='center', zorder=6)
for idx in [1, 2, 3, 4, 5, 6, 7]:
    ax2.annotate('', xy=(GCTR[idx][0], GCTR[idx][1]),
                 xytext=(qcx2, qcy2),
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=1.6,
                                 connectionstyle=f'arc3,rad={0.07+idx*0.07}'))
# Label the LAMP1→others arrow bundle as the negative pairs.
# Anchor near the midpoint of an outgoing arrow that's in clear space.
neg_target = GCTR[3]  # (2.2, 0.5) — uncluttered right-center region
ax2.text((qcx2 + neg_target[0]) / 2, (qcy2 + neg_target[1]) / 2,
         'neg\npairs', fontsize=32, ha='center', va='center',
         color='#888888', style='italic', zorder=7)

ft(BX + SUB_W / 2, TITLE_Y, 'Distinctiveness mAP',
   ha='center', fontsize=56, fontweight='bold', color=CDARK)
ft(BX + SUB_W / 2, SUBTITLE_Y,
   'pos: same geneKO reps\nneg: ALL other geneKOs',
   ha='center', va='center', fontsize=34, color=CGRAY, style='italic',
   linespacing=1.2)
ft(BX + SUB_W / 2, QUESTION_Y,
   'Is this geneKO distinct\nfrom all other geneKOs?',
   ha='center', va='center', fontsize=36, color=CDARK,
   linespacing=1.25,
   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=CFRAME,
             lw=1.5, alpha=0.95))

# ── A3: Consistency ───────────────────────────────────────────────────────
CX2 = PANEL_LEFTS[2]
ax3 = add_ax(CX2, SUB_Y, SUB_W, SUB_H)
scatter_base(ax3)
# Protein complexes (not pathways) — each cluster = the gene-KO
# replicates for member subunits of one complex. Two ribosomal
# subunits + the proteasome give a recognizable set with clearly
# different cellular roles for the schematic.
# Variable member counts per protein complex — real complexes have
# different subunit counts (large 60S ribo > 40S; 26S proteasome has
# ~33 subunits split across many KOs). Each entry = one member geneKO.
pathway_genes = {
    'Ribo60S\ngeneKOs':    {'col': CB,
        'pts': [(-2.1, 1.4), (-1.8, 1.1), (-2.3, 1.0), (-1.6, 1.7),
                (-2.5, 1.5), (-1.9, 0.85)]},          # 6 members
    'Ribo40S\ngeneKOs':    {'col': CG,
        'pts': [(1.6, 1.8), (1.9, 1.4), (1.4, 1.5),
                (2.0, 1.1)]},                          # 4 members
    'Proteasome\ngeneKOs': {'col': CR,
        'pts': [(-1.1, -1.6), (-0.8, -1.9), (-1.4, -1.2), (-0.9, -1.4),
                (-1.3, -1.85), (-0.6, -1.65), (-1.5, -1.55)]},  # 7 members
}
np.random.seed(7)
ctrs = {}
for pname, pdata in pathway_genes.items():
    col = pdata['col']
    spts = []
    for cx, cy in pdata['pts']:
        x = cx + np.random.randn() * 0.14
        y = cy + np.random.randn() * 0.14
        spts.append((x, y))
        ax3.scatter(x, y, c=col, s=260, alpha=0.92, zorder=4,
                    edgecolors=CDARK, linewidths=2.2)
    for ii in range(len(spts)):
        for jj in range(ii + 1, len(spts)):
            ax3.plot([spts[ii][0], spts[jj][0]],
                     [spts[ii][1], spts[jj][1]],
                     color=col, lw=1.8, alpha=0.45, zorder=3)
    xs = [p[0] for p in spts]; ys = [p[1] for p in spts]
    # Size the dashed cluster ring from the actual point spread plus a
    # margin, so it always encompasses every dot regardless of the
    # per-seed jitter. The blue cluster was tight; green/red landed a
    # bit wider after the s=260 marker bump, leaving outer dots
    # poking past the original fixed 1.3×0.85 ellipse.
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    e_w = max(1.3, span_x + 0.7)
    e_h = max(0.85, span_y + 0.6)
    ax3.add_patch(Ellipse((np.mean(xs), np.mean(ys)), e_w, e_h, angle=10,
                           fill=False, edgecolor=col, lw=3.2,
                           linestyle='--', alpha=0.85, zorder=5))
    ctrs[pname] = (np.mean(xs), np.mean(ys))
    # Push pathway labels well clear of the cluster + connector lines.
    # Blue (Trafficking) and green (Metabolism) sit ABOVE their clusters
    # at a higher y so they don't overlap the ellipse / line tangle;
    # red (Translation) sits BELOW + to the right so it doesn't collide
    # with the inter-pathway `neg pairs` arrow.
    offsets = {'Ribo60S\ngeneKOs':    (0.0, 1.30),
               'Ribo40S\ngeneKOs':    (0.0, 1.20),   # slightly higher
               'Proteasome\ngeneKOs': (1.40, -1.25)}  # slightly lower
    dx, dy = offsets.get(pname, (0, 0.7))
    ax3.text(np.mean(xs) + dx, np.mean(ys) + dy, pname,
             fontsize=36, color=col, fontweight='bold',
             ha='center', va='center', zorder=6)
for x, y, i in PTS:
    if CATMAP.get(i, 'Other') not in ('Traf', 'Meta', 'Trans'):
        ax3.scatter(x, y, c='#bbbbbb', s=70, alpha=0.30, zorder=2,
                    edgecolors='none')
pnames = list(ctrs.keys())
# Blue-anchored neg-pair arrows (Ribo60S → Ribo40S, Ribo60S → Proteasome)
# match Plot A and Plot B's convention of drawing neg pairs only from
# the focal (blue) cluster. Even though mAP technically uses ALL pairs,
# the schematic is clearer when readers see one consistent "focal
# example" across all three panels (LAMP1 in A/B, Ribo60S in C).
for ia, ib in [(0, 1), (0, 2)]:
    pa = ctrs[pnames[ia]]; pb = ctrs[pnames[ib]]
    mx = (pa[0] + pb[0]) / 2; my = (pa[1] + pb[1]) / 2
    ax3.annotate('', xy=pb, xytext=pa,
                 arrowprops=dict(arrowstyle='<->', color='#888888', lw=2.0,
                                 connectionstyle='arc3,rad=0.15'))
    ax3.text(mx, my, 'neg\npairs', fontsize=32, ha='center', va='center',
             color='#888888', style='italic')

ft(CX2 + SUB_W / 2, TITLE_Y, 'Consistency mAP',
   ha='center', fontsize=56, fontweight='bold', color=CDARK)
ft(CX2 + SUB_W / 2, SUBTITLE_Y,
   'pos: same protein complex\nneg: ALL other geneKOs',
   ha='center', va='center', fontsize=34, color=CGRAY, style='italic',
   linespacing=1.2)
ft(CX2 + SUB_W / 2, QUESTION_Y,
   'Do protein complex members\ncluster together?',
   ha='center', va='center', fontsize=36, color=CDARK,
   linespacing=1.25,
   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=CFRAME,
             lw=1.5, alpha=0.95))

import os
out_png = os.path.join(os.path.dirname(__file__),
                       'distinctiveness_map_schematic_panelA.png')
out_pdf = os.path.join(os.path.dirname(__file__),
                       'distinctiveness_map_schematic_panelA.pdf')
# `bbox_inches='tight'` crops to the smallest bbox containing all
# artists, but for fig.text with a rounded bbox the artist's
# bbox-patch is sometimes missed and the bottom of the question box
# gets clipped. `pad_inches=0.5` forces a half-inch white margin
# around the tight bbox so nothing rides the edge.
fig.savefig(out_png, dpi=160, bbox_inches='tight', pad_inches=0.5,
             facecolor='white')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.5,
             facecolor='white')
print(f'Saved: {out_png}')
print(f'       {out_pdf}')

if __name__ == '__main__':
    pass
