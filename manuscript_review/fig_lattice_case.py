"""Case-study figure: the admissible plans of one node installed in 2025 (single panel).

Geometry from the case-study parameters (Section 3.5): lattice units 0.25 MVA / 0.5 MWh, duration 2-4 h,
at most 5 MWh per node, budget 1 MEUR at the 2025 expected unit costs (256,317.32 EUR/MVA, 253,877.68 EUR/MWh,
W167). Panel (b) is an illustrative poll, not a recorded one.
"""
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = sys.argv[1]

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

DS, DE = 0.25, 0.5                # lattice units (MVA, MWh)
PHI_MIN, PHI_MAX = 2.0, 4.0       # duration bounds (h)
EMAX = 5.0                        # MWh per node
CS, CE, BUDGET = 256.31732, 253.87768, 1000.0   # kEUR/MVA, kEUR/MWh, kEUR (2025, expected)

def classify(S, E):
    if S == 0 and E == 0:
        return "zero"
    if S == 0 or not (PHI_MIN * S - 1e-9 <= E <= PHI_MAX * S + 1e-9) or E > EMAX + 1e-9:
        return "out"
    return "adm" if CS * S + CE * E <= BUDGET + 1e-9 else "budget"

fig, ax = plt.subplots(1, 1, figsize=(90 / 25.4, 92 / 25.4))

# ---------------- (a) the lattice of one node, 2025 ----------------
Smax, Emax_plot = 2.75, 5.5
pts = {"adm": [], "budget": [], "out": []}
for i in range(int(Smax / DS) + 1):
    for j in range(int(Emax_plot / DE) + 1):
        S, E = i * DS, j * DE
        c = classify(S, E)
        if c != "zero":
            pts[c].append((S, E))
s = [0, Smax]
ax.plot(s, [PHI_MIN * x for x in s], ls="--", lw=0.8, color=INK2, zorder=1)
ax.plot(s, [PHI_MAX * x for x in s], ls="--", lw=0.8, color=INK2, zorder=1)
ax.axhline(EMAX, ls=":", lw=0.9, color=INK2, zorder=1)
ax.plot(s, [(BUDGET - CS * x) / CE for x in s], ls="-", lw=1.0, color=INK, zorder=1)
ax.scatter(*zip(*pts["out"]), s=3, color=GRID, zorder=2)
ax.scatter(*zip(*pts["budget"]), s=22, facecolors="white", edgecolors=ORANGE, linewidths=1.0, zorder=3)
ax.scatter(*zip(*pts["adm"]), s=22, color=BLUE, edgecolors="white", linewidths=0.6, zorder=4)
ax.scatter([0], [0], marker="D", s=26, color=INK, zorder=5)
ax.annotate("0.25 MVA / 0.5 MWh\n(smallest unit, 2 h)", (0.25, 0.5), (0.62, 0.12), fontsize=6.8, color=INK2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
ax.annotate("0.25 MVA / 1 MWh\n(reference unit, 4 h)", (0.25, 1.0), (0.03, 4.45), fontsize=6.8, color=INK2,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
ax.text(1.42, 5.2, "4 h", fontsize=7, color=INK2)
ax.text(2.58, 4.55, "2 h", fontsize=7, color=INK2)
ax.set_xlim(-0.05, Smax + 0.05); ax.set_ylim(-0.12, Emax_plot + 0.1)
ax.set_xticks([i * 0.5 for i in range(6)]); ax.set_yticks(range(6))
ax.set_xlabel("converter rating $S^{\\mathrm{Inv}}$ (MVA)")
ax.set_ylabel("energy capacity $E^{\\mathrm{Inv}}$ (MWh)")
leg = [Line2D([], [], marker="D", ls="", color=INK, markersize=4.5, label="no storage ($x = 0$)"),
       Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", markersize=5.2, label="admissible plan"),
       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ORANGE, markersize=5.2,
              label="within limits, over budget"),
       Line2D([], [], marker="o", ls="", color=GRID, markersize=2.5, label="outside the limits"),
       Line2D([], [], ls="--", lw=0.8, color=INK2, label="duration bounds"),
       Line2D([], [], ls=":", lw=0.9, color=INK2, label="$E^{\\mathrm{Max}}$"),
       Line2D([], [], ls="-", lw=1.0, color=INK, label="budget (single node, 2025)")]
fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, handletextpad=0.35, columnspacing=1.0,
           labelspacing=0.35, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
fig.tight_layout(rect=(0, 0.2, 1, 1))
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200)
print("adm", len(pts["adm"]), "over budget", len(pts["budget"]))
