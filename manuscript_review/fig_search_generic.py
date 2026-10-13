"""Generic schematic of the planning search (mesh adaptive direct search on discrete plan sizes).

Toy illustration, no case-study numbers: a grid of discrete plans (power units x energy units), an admissible
region (duration bounds i <= j <= 2i and a budget i + j <= 9), an illustrative cost F with its minimum near
(3.2, 4.4), and four polls that follow the rules of Algorithm 1, variant A: success at step 2 (step doubles to 4),
failure at step 4 (halves to 2), failure at step 2 (halves to 1), complete unit poll fails (stop).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle

OUT = sys.argv[1]
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7.2,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#cfcec9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
N = 9

def admissible(i, j):
    return (i == 0 and j == 0) or (i >= 1 and i <= j <= 2 * i and i + j <= 9)

def F(i, j):
    return (i - 4.0) ** 2 + 0.6 * (j - 5.2) ** 2 + 0.3 * (i - 4.0) * (j - 5.2)

polls = [  # (incumbent, step, directions, title)
    ((2, 3), 2, [(1, 1), (-1, 1), (0, -1)], "(a) $\\Delta = 2$: a better plan is\nfound and the step doubles"),
    ((4, 5), 4, [(1, 1), (-1, 0), (0, -1)], "(b) $\\Delta = 4$: no candidate is\nadmissible; the step is halved"),
    ((4, 5), 2, [(-1, -1), (1, 0), (0, 1)], "(c) $\\Delta = 2$: no candidate is\nbetter; the step is halved"),
    ((4, 5), 1, None, "(d) $\\Delta = 1$: no neighbouring\nplan is better: stop"),
]

fig, axes = plt.subplots(1, 4, figsize=(190 / 25.4, 70 / 25.4))
region = Polygon([(0, 0), (3, 6), (4.5, 4.5)], closed=True, facecolor=BLUE, alpha=0.10, edgecolor="none", zorder=0)
xx, yy = np.meshgrid(np.linspace(-0.5, N + 0.5, 200), np.linspace(-0.5, N + 0.5, 200))
for k, (ax, (inc, step, dirs, title)) in enumerate(zip(axes, polls)):
    ax.add_patch(Polygon(region.get_xy(), closed=True, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=0.6, zorder=0))
    ax.contour(xx, yy, F(xx, yy), levels=[0.6, 2, 4.5, 8, 13, 19], colors=GRID, linewidths=0.5, zorder=1)
    for i in range(N + 1):
        for j in range(N + 1):
            ax.scatter(i, j, s=5 if admissible(i, j) else 2.5, color=BLUE if admissible(i, j) else GRID,
                       zorder=2, edgecolors="none")
    if dirs is None:
        cand = [(inc[0] + di, inc[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        ax.add_patch(Rectangle((inc[0] - 1, inc[1] - 1), 2, 2, fill=False, ls=(0, (3, 2)), lw=0.8, edgecolor=INK2,
                               zorder=2))
    else:
        cand = [(inc[0] + step * d[0], inc[1] + step * d[1]) for d in dirs]
    best = None
    for c in cand:
        if dirs is not None:
            ax.add_patch(FancyArrowPatch(inc, c, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK2,
                                         shrinkA=4, shrinkB=4, zorder=3))
        if not admissible(*c):
            ax.scatter(*c, marker="x", s=24, color=ORANGE, linewidths=1.2, zorder=5, clip_on=False)
        elif F(*c) < F(*inc):
            ax.scatter(*c, s=34, color=AQUA, edgecolors="white", linewidths=0.6, zorder=5)
            best = c
        else:
            ax.scatter(*c, s=28, facecolors="white", edgecolors=ORANGE, linewidths=1.2, zorder=5)
    ax.scatter(*inc, marker="*", s=90, color=INK, edgecolors="white", linewidths=0.5, zorder=6)
    if best is not None:
        ax.annotate("", best, inc, arrowprops=dict(arrowstyle="-|>", lw=1.4, color=AQUA, shrinkA=5, shrinkB=5),
                    zorder=4)
    ax.scatter(0, 0, marker="D", s=16, color=INK, zorder=6)
    ax.set_xlim(-0.6, N + 0.4); ax.set_ylim(-0.6, N + 0.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("power rating", fontsize=7.5, labelpad=2)
    if k == 0:
        ax.set_ylabel("energy capacity", fontsize=7.5, labelpad=2)
    ax.set_title(title, loc="left", fontsize=7.6)

leg = [Line2D([], [], marker="s", ls="", markerfacecolor=BLUE, alpha=0.25, markeredgecolor=BLUE, markersize=7,
              label="admissible plans (size, duration, budget limits)"),
       Line2D([], [], marker="o", ls="", color=GRID, markersize=2.5, label="plan outside the limits"),
       Line2D([], [], marker="D", ls="", color=INK, markersize=3.8, label="no storage"),
       Line2D([], [], ls="-", lw=0.5, color=GRID, label="total cost $F$ (illustrative)"),
       Line2D([], [], marker="*", ls="", color=INK, markersize=8, label="current plan"),
       Line2D([], [], marker="o", ls="", color=AQUA, markeredgecolor="white", markersize=5.5, label="candidate, better"),
       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.2,
              markersize=5, label="candidate, not better"),
       Line2D([], [], marker="x", ls="", color=ORANGE, markersize=5, markeredgewidth=1.2,
              label="candidate, not admissible")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, handletextpad=0.35, columnspacing=1.2,
           labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=0.6, rect=(0, 0.15, 1, 0.99))
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200)
for inc, step, dirs, _ in polls:
    if dirs is None:
        cand = [(inc[0] + di, inc[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    else:
        cand = [(inc[0] + step * d[0], inc[1] + step * d[1]) for d in dirs]
    print(inc, step, [(c, "adm" if admissible(*c) else "out", round(F(*c) - F(*inc), 2) if admissible(*c) else None)
                      for c in cand])
