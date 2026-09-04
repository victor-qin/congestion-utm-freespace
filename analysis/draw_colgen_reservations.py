"""Draw how colgen files airspace reservations today, against the symmetric-pad filing it replaced.

Three panels, all measured from the repo's own geometry (`corridor_segment_volume`,
`derive_cell_window`, `hex_center`) rather than sketched:

1. Plan view: the hex lattice, a 4-cell column path, and the oriented transit boxes the
   ledger actually receives (pitch-long, extended half a width beyond each end).
2. Space-time view of the LEGACY symmetric time pads: every hop box was live
   ``[t0 - buf, t1 + buf]``, so a cell's two incident hops smeared its capacity-row
   footprint to 4 periods ``(-2, +1)``.
3. The same trajectory as shipped TODAY, trailing pad moved onto the leading end
   (``[t0, t1 + buf]``): the footprint aligns with the grid and shrinks to 3 periods
   ``(-1, +1)``, which is what drops ``revisit_depth`` to 2 (issue #94, refinement 2).
   Panel 3 is the arm the guard below pins against the live config; panel 2 is drawn from
   its own pad arguments and is retained purely as the before-picture.

    uv run python analysis/draw_colgen_reservations.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen.windows import derive_cell_window  # noqa: E402
from freespace_sim.planner.hexgrid import circumradius, hex_center  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

# dataviz reference palette (light mode): slots 1-3 + ink/neutral tokens.
BLUE = "#2a78d6"      # reservation volumes (both filings)
ORANGE = "#eb6834"    # the leading pad / what moved
AQUA = "#1baf7a"      # derived capacity-row claims
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"

CELLS = [(0, 0), (1, 0), (2, 0), (2, 1)]  # one 60-degree turn, hex-adjacent throughout


def hex_vertices(center: np.ndarray, radius: float, neighbor_angle: float) -> np.ndarray:
    """Vertices of a lattice hex: edges face the neighbours, so vertices sit 30 deg off."""

    angles = neighbor_angle + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], 1)


def oriented_box(p0: np.ndarray, p1: np.ndarray, half_w: float, ext: float) -> np.ndarray:
    """Plan-view corners of `corridor_segment_volume`'s box: extended, then widened."""

    direction = p1 - p0
    u = direction / np.linalg.norm(direction)
    n = np.array([-u[1], u[0]])
    a, b = p0 - u * ext, p1 + u * ext
    return np.stack([a + n * half_w, b + n * half_w, b - n * half_w, a - n * half_w])


def main() -> None:
    cfg = get_scenario("colgen_test").config()
    radius = circumradius(cfg)
    dt, buf = cfg.dt_s, cfg.time_buffer_s
    pitch = cfg.corridor_segment_len_m
    half_w = cfg.corridor_width_m / 2.0
    offsets = derive_cell_window(cfg)
    if offsets != (-1, 1):
        raise SystemExit(f"expected the shipped (-1, 1) window, measured {offsets}")

    centers = [np.array(hex_center(q, r, radius), dtype=float) for q, r in CELLS]
    nb = centers[1] - centers[0]
    neighbor_angle = math.atan2(nb[1], nb[0])

    fig = plt.figure(figsize=(13.5, 10.5), facecolor=SURFACE)
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.44, 0.56), left=0.06, right=0.985, top=0.93, bottom=0.075,
        hspace=0.24, wspace=0.16,
    )
    ax_plan = fig.add_subplot(grid[0, :])
    ax_sym = fig.add_subplot(grid[1, 0])
    ax_asym = fig.add_subplot(grid[1, 1], sharey=ax_sym)
    fig.suptitle(
        "How colgen files airspace reservations on the ledger "
        "(measured from corridor_segment_volume / derive_cell_window)",
        fontsize=13, color=INK, y=0.975,
    )

    # ------------------------------------------------------------------ plan view
    ax_plan.set_facecolor(SURFACE)
    ring = {c for c in CELLS}
    for q, r in CELLS:
        for dq in range(-1, 3):
            for dr in range(-1, 2):
                ring.add((q + dq, r + dr))
    for cell in sorted(ring):
        c = np.array(hex_center(*cell, radius))
        face = "#f1f0ec" if cell in CELLS else SURFACE
        ax_plan.add_patch(
            Polygon(hex_vertices(c, radius, neighbor_angle), closed=True, facecolor=face,
                    edgecolor=GRID, linewidth=1.0, zorder=1)
        )
    label_offsets = [(-46, 40), (-14, -52), (30, -46), (34, 6)]
    for i, (cell, c) in enumerate(zip(CELLS, centers)):
        ax_plan.plot(*c, "o", color=INK2, markersize=4, zorder=5)
        ax_plan.annotate(f"{chr(65 + i)}  (v={i})", c + np.array(label_offsets[i]),
                         color=INK, fontsize=9.5, zorder=6)
    for p0, p1 in zip(centers, centers[1:]):
        ax_plan.add_patch(
            Polygon(oriented_box(p0, p1, half_w, half_w), closed=True, facecolor=BLUE,
                    alpha=0.30, edgecolor=BLUE, linewidth=1.6, zorder=3)
        )
    ax_plan.add_patch(Circle(centers[0], cfg.effective_hover_radius_m, fill=False,
                             edgecolor=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2))
    ax_plan.annotate("endpoint hover cylinder\n(exact dwell window, separate claim rule)",
                     centers[0] + np.array([-150, -108]), color=INK2, fontsize=8)
    mid = 0.5 * (centers[1] + centers[2])
    ax_plan.annotate(
        f"one Volume4D per hop: box spans centre→centre (pitch {pitch:.0f} m),\n"
        f"width {cfg.corridor_width_m:.0f} m, extended +{half_w:.0f} m past each end "
        "(ASTM contiguity)\nso consecutive boxes overlap in space",
        (mid[0] - 40, mid[1] - 205), color=INK, fontsize=9,
    )
    arrow = FancyArrowPatch(centers[0] + 18, centers[3] - 18, arrowstyle="-|>",
                            mutation_scale=16, color=INK, linewidth=1.4, zorder=6)
    ax_plan.add_patch(arrow)
    ax_plan.set_title("Plan view: a column's cell path becomes one oriented box per lattice hop",
                      fontsize=11, color=INK, loc="left")
    ax_plan.set_aspect("equal")
    ax_plan.set_xlim(-215, 505)
    ax_plan.set_ylim(-230, 175)
    ax_plan.set_xlabel("east (m)", fontsize=8, color=INK2)
    ax_plan.tick_params(labelsize=8, colors=INK2)
    for spine in ax_plan.spines.values():
        spine.set_color(GRID)

    # ------------------------------------------------- space-time panels (unrolled)
    s = [i * pitch for i in range(len(CELLS))]  # arc-length of each visit
    focus = 2  # cell C

    def draw_spacetime(ax, trailing_pad: float, leading_pad: float, title: str) -> None:
        ax.set_facecolor(SURFACE)
        t_max = 4 * dt + buf + 2.0
        for cell_i in range(len(CELLS)):  # cell bands
            ax.axvline(s[cell_i] - pitch / 2, color=GRID, linewidth=0.8, zorder=1)
        ax.axvline(s[-1] + pitch / 2, color=GRID, linewidth=0.8, zorder=1)
        for j in range(-2, int(t_max // dt) + 1):  # period grid
            ax.axhline(j * dt, color=GRID, linewidth=0.8, zorder=1)
            if j * dt <= t_max - dt:
                ax.annotate(f"v={j}", (s[0] - pitch / 2 - 8, j * dt + dt / 2), fontsize=8,
                            color=INK2, ha="right", va="center")

        # Row claims of the focus cell: union of periods its incident hop boxes touch.
        lo = math.floor((focus * dt - dt - trailing_pad) / dt)
        hi = math.ceil(((focus + 1) * dt + leading_pad) / dt) - 1
        band = (s[focus] - pitch / 2, s[focus] + pitch / 2)
        ax.fill_betweenx([lo * dt, (hi + 1) * dt], band[0], band[1], color=AQUA, alpha=0.18,
                         zorder=2)
        for j in range(lo, hi + 1):
            ax.fill_betweenx([j * dt + 0.35, (j + 1) * dt - 0.35], band[0] + 2, band[1] - 2,
                             facecolor="none", edgecolor=AQUA, linewidth=1.4, zorder=4)
        n_periods = hi - lo + 1
        ax.annotate(
            f"cell C claims rows v={lo}..{hi}\n({n_periods} periods, offsets ({lo - focus},"
            f" {hi - focus}))  →  revisit_depth = {hi - lo}",
            (s[focus], 4 * dt + buf + 1.4 * dt), color=INK, fontsize=9, ha="center",
        )

        # One space-time rectangle per hop box: full 2-cell span for its whole window.
        for i in range(len(CELLS) - 1):
            x0, x1 = s[i] - half_w, s[i + 1] + half_w
            t0, t1 = i * dt, (i + 1) * dt
            ax.fill_betweenx([t0 - trailing_pad, t1 + leading_pad], x0, x1, color=BLUE,
                             alpha=0.30, edgecolor=BLUE, linewidth=1.4, zorder=3)
            ax.fill_betweenx([t1, t1 + leading_pad], x0, x1, color=ORANGE, alpha=0.42,
                             edgecolor="none", zorder=3)
            if trailing_pad == 0.0:
                ax.fill_betweenx([t0 - buf, t0], x0, x1, facecolor="none", edgecolor=INK2,
                                 hatch="///", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.5,
                                 zorder=2)
        ax.plot(s, [i * dt for i in range(len(CELLS))], "-o", color=INK, linewidth=1.6,
                markersize=4.5, zorder=5)
        for i, name in enumerate("ABCD"):
            ax.annotate(name, (s[i], -2.0 * dt - 2.6), ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(s[0] - pitch / 2 - 46, s[-1] + pitch / 2 + 12)
        ax.set_ylim(-2 * dt - 4.6, t_max + 2.6 * dt)
        ax.set_xlabel("distance along path (m); bands = hex cells", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    draw_spacetime(
        ax_sym, buf, buf,
        f"Legacy (replaced): symmetric time pads  [t₀−{buf:.0f}, t₁+{buf:.0f}] s "
        f"(buf = dt = {dt:.0f} s)",
    )
    draw_spacetime(
        ax_asym, 0.0, buf,
        f"Today: trailing end on the grid, pad in front  [t₀, t₁+{buf:.0f}] s",
    )
    ax_sym.set_ylabel("time (s); horizontal lines every dt", fontsize=8.5, color=INK2)
    top_y = 4 * dt + buf + 2.55 * dt
    ax_sym.annotate("enforced gap between two conflicting transits\n= trailing + leading pad "
                    f"= {2 * buf:.0f} s", (s[0] - pitch / 2 + 4, top_y),
                    fontsize=9, color=INK, va="top")
    ax_asym.annotate(f"enforced gap = 0 + leading pad = {buf:.0f} s (one dt)\n"
                     "hatched = where the trailing pad was",
                     (s[0] - pitch / 2 + 4, top_y), fontsize=9, color=INK, va="top")
    # Name both pads where the margins are free: the first hop's trailing pad on the
    # left, the last hop's leading pad on the right.
    ax_sym.annotate("trailing pad", (s[1] + half_w + 6, -0.5 * buf), fontsize=8, color=INK2,
                    va="center")
    ax_sym.annotate("leading pad", (s[-1] + half_w + 5, 3 * dt + 0.5 * buf), fontsize=8,
                    color=INK2, va="center")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.30, edgecolor=BLUE),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, alpha=0.42),
        plt.Rectangle((0, 0), 1, 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA),
        plt.Line2D([], [], color=INK, marker="o", markersize=4.5, linewidth=1.6),
    ]
    fig.legend(handles,
               ["transit Volume4D (hop box, live window)",
                "leading time pad (time_buffer_s)",
                "capacity rows the visit claims (visit_rows)",
                "trajectory: centre reached at step v"],
               loc="lower center", ncols=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))

    out = REPO_ROOT / "analysis" / "colgen_reservation_filing.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
