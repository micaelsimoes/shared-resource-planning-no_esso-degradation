"""Redraw of Fig. res_ageing with the calibration names of Table res_ageing (the frozen figure used code labels).

Numbers are those of Table res_ageing (value - I, threshold, first binding year of the 0.70 floor); style follows the
frozen figure: Arial-metric font, Okabe-Ito blue/vermillion, +/- threshold bars, zero line, floor column on the right.
"""
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = sys.argv[1]
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.it": "Liberation Sans:italic", "mathtext.rm": "Liberation Sans", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BLUE, VERM, GREY = "#0072b2", "#d55e00", "#999999"

rows = [  # label, value - I (kEUR), threshold (kEUR), determinate?, binding year of the 0.70 floor
    ("Baseline (cycling + calendar)", -64.4, 13.1, True, "2035"),
    ("Cycling only", -31.6, 12.6, True, "never"),
    ("Retention 50 %", -73.7, 12.8, True, "2035"),
    ("Retention 50 %, mid-block", -65.9, 12.9, True, "2035"),
    ("Datasheet 8,000 cycles", -45.2, 12.6, True, "never"),
    ("No ageing", -4.1, 12.9, False, "n/a"),
]

fig, ax = plt.subplots(figsize=(4.3, 2.55))
n = len(rows)
for k, (lab, v, thr, det, yr) in enumerate(rows):
    y = n - 1 - k
    col = BLUE if det else VERM
    ax.errorbar(v, y, xerr=thr, fmt="o", color=col, ms=4.5, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
    ax.text(v, y + 0.27, f"−{abs(v):.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.text(1.03, y, yr, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.6)
ax.text(1.03, n - 0.35, "0.70 floor\nbinds", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=7.2, style="italic")
ax.axvline(0, color=GREY, lw=0.8, zorder=1)
ax.set_yticks(range(n))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_ylim(-1.35, n - 0.35)
ax.set_xlim(-92, 14)
ax.set_xticks([-80, -60, -40, -20, 0])
ax.set_xticklabels(["−80", "−60", "−40", "−20", "0"])
ax.set_xlabel("value − $I$ of the reference unit (k€)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
leg = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="determinate (± threshold)"),
       Line2D([], [], marker="o", ls="", color=VERM, ms=4.5, label="within resolution")]
ax.legend(handles=leg, loc="lower left", ncol=2, frameon=False, fontsize=7.2, handletextpad=0.2, columnspacing=1.0,
          bbox_to_anchor=(-0.02, -0.02))
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=220)
