"""Generate every ``context/figures/*.png`` schematic cited from ``freespace_sim`` docstrings.

Each ``fig_*`` renders one PNG illustrating a spatial concept that a docstring/comment cites in
place of describing it in prose (AGENTS.md: "make a PNG ... and cite that figure as a comment").
Schematic and not to scale; constants match the shipped ``SimConfig`` defaults (terminal_radius
90 m, corridor 60 x 30 m, ceiling 125 m, hex ``x = R√3(q + r/2)``, ``y = 1.5R·r``). ``context/`` is
unmaintained (AGENTS.md), so this is a single throwaway generator, not per-figure modules.

Run: ``uv run python context/figures/make_figures.py``   (writes every PNG beside this file)
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, RegularPolygon,
)

from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.geometry import BoxSpec  # noqa: E402
from freespace_sim.planner import hexgrid as hg  # noqa: E402
from freespace_sim.planner.colgen import translate  # noqa: E402
from freespace_sim.planner.colgen.network import (  # noqa: E402
    RowKey, _column_endpoint_steps, build_flight_graph, column_claims,
)
from freespace_sim.planner.colgen.objective import cost_model  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.windows import derive_cell_window  # noqa: E402
from freespace_sim.types import FlightRequest, IntentStatus, Terminal  # noqa: E402

OUT = "context/figures"
SQRT3 = 3.0 ** 0.5

# Shared print-safe palette.
INK = "#2d3748"
GRID = "#9aa5b1"
HEXGRID = "#cbd5e0"
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
RED = "#c53030"
GREEN = "#2f855a"
RAW = "#a0aec0"


def _save(fig, name: str) -> None:
    """Write ``fig`` to ``context/figures/<name>.png`` at 150 dpi and close it."""
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def _axial_to_xy(q: float, r: float, R: float) -> tuple[float, float]:
    """World (x, y) of pointy-top axial hex ``(q, r)``: ``x = R√3(q + r/2)``, ``y = 1.5R·r``."""
    return (R * SQRT3 * (q + r / 2.0), R * 1.5 * r)


def _edge_point(center, toward, r: float) -> tuple[float, float]:
    """The radius-``r`` point on ``center`` in the direction of ``toward`` (xy)."""
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    n = (dx * dx + dy * dy) ** 0.5
    return (center[0] + r * dx / n, center[1] + r * dy / n)


def _hex_dist(a, b) -> int:
    """Axial hex distance between ``(q, r)`` cells (cube metric)."""
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(ar - br) + abs(aq + ar - bq - br)) // 2


def _box(ax, x, y, text, *, fc="white", ec=INK, tc=INK, fs=8.5, pad=0.5) -> None:
    """Rounded text box centred at ``(x, y)`` — the shared flow-chart primitive."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5,
            bbox=dict(boxstyle=f"round,pad={pad}", facecolor=fc, edgecolor=ec, linewidth=1.5))


def fig_enroute_rulers() -> None:
    """volumes: the reference / flown / detour en-route rulers."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ox, dx, r = 0.0, 10.0, 1.0
    for cx, name in ((ox, "ORIGIN HUB"), (dx, "DEST HUB")):
        ax.add_patch(Circle((cx, 0), r, facecolor="#edf2f7", edgecolor=GRID, lw=1.4, zorder=1))
        ax.plot([cx], [0], "o", color=INK, ms=5, zorder=4)
        ax.text(cx, -1.55, name, ha="center", va="top", fontsize=9, color=INK, weight="bold")
    yc = 2.15
    ax.annotate("", xy=(dx, yc), xytext=(ox, yc),
                arrowprops=dict(arrowstyle="<->", color=GRID, lw=1.2))
    for cx in (ox, dx):
        ax.plot([cx, cx], [0, yc], ls=":", color=GRID, lw=0.9, zorder=0)
    ax.text((ox + dx) / 2, yc + 0.12, "centre → centre  =  5385 m", ha="center", va="bottom",
            fontsize=9, color=GRID)
    ax.annotate("", xy=(ox + r, 0), xytext=(ox, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.text(ox + r / 2, 0.16, "r_o = 210 m", ha="center", fontsize=8, color=INK)
    ax.annotate("", xy=(dx - r, 0), xytext=(dx, 0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.text(dx - r / 2, 0.16, "r_d = 210 m", ha="center", fontsize=8, color=INK)
    ax.plot([ox + r, dx - r], [0, 0], color=BLUE, lw=3.0, solid_capstyle="round", zorder=3)
    ax.text((ox + dx) / 2, -0.42, "reference = 5385 − 210 − 210 = 4965 m  (edge→edge straight)",
            ha="center", va="top", fontsize=9, color=BLUE, weight="bold")
    ax.plot([ox + r, 2.2, 4.0, 5.6, 7.4, dx - r], [0.0, 1.05, 1.5, 1.42, 0.95, 0.0], color=ORANGE,
            lw=2.4, marker="o", ms=4, zorder=3)
    ax.text((ox + dx) / 2, 1.72, "flown = Σ segments = 5689 m  (actual folded path)", ha="center",
            va="bottom", fontsize=9, color=ORANGE, weight="bold")
    ax.text((ox + dx) / 2, -0.95, "detour = max(0, flown − reference) = 724 m", ha="center",
            va="top", fontsize=9, color=INK)
    ax.set_title("En-route rulers — both start and end on the exit_radius circles,\n"
                 "so their difference is pure en-route detour (nothing inside a hub column counts)",
                 fontsize=10.5, color=INK, pad=10)
    ax.set_xlim(-2.0, 12.0)
    ax.set_ylim(-2.6, 2.9)
    _save(fig, "enroute_rulers")


def fig_corridor_box_extension() -> None:
    """volumes.corridor_segment_volume: anisotropic width/2 (level) vs height/2 (climb) extension."""
    w, h = 60.0, 30.0
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axp.set_title("Top-down: level segment — extend by corridor_width/2", fontsize=10, color=INK)
    ext = w / 2
    a, b, c = 120.0, 300.0, 480.0
    axp.add_patch(Rectangle((a - ext, -w / 2), (b + ext) - (a - ext), w, facecolor="#ebf3fb",
                            edgecolor=BLUE, lw=1.8))
    axp.add_patch(Rectangle((b - ext, -w / 2 - 4), (c + ext) - (b - ext), w, facecolor="none",
                            edgecolor=ORANGE, lw=1.8, ls="--"))
    axp.plot([a, b], [0, 0], color=INK, lw=1.5, marker="o", ms=4)
    axp.plot([b, c], [-4, -4], color=ORANGE, lw=1.2, marker="o", ms=3)
    axp.annotate("", xy=(a, w / 2 + 14), xytext=(a - ext, w / 2 + 14),
                 arrowprops=dict(arrowstyle="<->", color=GRID, lw=1.0))
    axp.text(a - ext / 2, w / 2 + 18, "width/2", ha="center", fontsize=8, color=GRID)
    axp.annotate("", xy=(b + ext, -w / 2 - 16), xytext=(b + ext, w / 2),
                 arrowprops=dict(arrowstyle="<->", color=GRID, lw=1.0))
    axp.text(b + ext + 10, 0, "width\n(= half-width\neach side)", ha="left", va="center",
             fontsize=8, color=GRID)
    axp.text((a + c) / 2, -w / 2 - 34, "consecutive boxes overlap by width/2 → ASTM §4.3.5 contiguity",
             ha="center", fontsize=8.5, color=INK)
    axp.set_xlim(a - ext - 30, c + ext + 90)
    axp.set_ylim(-w / 2 - 50, w / 2 + 34)
    axp.set_aspect("equal")
    axp.axis("off")

    axs.set_title("Side view: climb segment — extend by corridor_height/2 (not width/2)",
                  fontsize=10, color=INK)
    ceiling, z0, z1, xc = 125.0, 30.0, 110.0, 0.0
    axs.add_patch(Rectangle((xc - w / 2, z0 - h / 2), w, (z1 + h / 2) - (z0 - h / 2),
                            facecolor="#ebf3fb", edgecolor=BLUE, lw=1.8))
    axs.add_patch(Rectangle((xc - w / 2, z0 - w / 2), w, (z1 + w / 2) - (z0 - w / 2),
                            facecolor="none", edgecolor=RED, lw=1.5, ls=":"))
    axs.plot([xc, xc], [z0, z1], color=INK, lw=1.6, marker="o", ms=4)
    axs.axhline(ceiling, color=RED, lw=1.2, ls="-")
    axs.text(w / 2 + 6, ceiling, "airspace_ceiling_m", va="center", fontsize=8, color=RED)
    axs.text(w / 2 + 6, (z0 + z1) / 2, "correct: z ∈ [z0, z1] ± height/2", va="center", fontsize=8,
             color=BLUE)
    axs.text(w / 2 + 6, z1 + w / 2 + 4, "width/2 would overshoot →", va="bottom", fontsize=8,
             color=RED)
    axs.set_xlim(-w / 2 - 10, w / 2 + 150)
    axs.set_ylim(0, ceiling + 40)
    axs.set_aspect("equal")
    axs.axis("off")

    fig.suptitle("corridor_segment_volume: anisotropic longitudinal extension", fontsize=11.5,
                 color=INK, y=1.0)
    _save(fig, "corridor_box_extension")


def fig_exit_radius() -> None:
    """volumes.exit_radius: exit-lane inner edge vs the hub column, overlap penetration/gap."""
    r_t, w = 90.0, 60.0
    r_e = r_t + w / 2
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0], [0], "o", color=INK, ms=6, zorder=5)
    ax.text(0, -12, "hub centre", ha="center", va="top", fontsize=9, color=INK)
    ax.add_patch(Circle((0, 0), r_t, facecolor="#edf2f7", edgecolor=BLUE, lw=1.8, zorder=1))
    ax.add_patch(Circle((0, 0), r_e, facecolor="none", edgecolor=ORANGE, lw=1.8, ls="--", zorder=2))
    ax.annotate("", xy=(r_t * 0.71, r_t * 0.71), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.text(r_t * 0.36, r_t * 0.40, "terminal_radius\n= 90 m", color=BLUE, fontsize=8.5, ha="center")
    ax.annotate("", xy=(-r_e * 0.71, r_e * 0.71), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.text(-r_e * 0.5, r_e * 0.78, "exit_radius\n= 90 + 60/2 = 120 m", color=ORANGE, fontsize=8.5,
            ha="center")
    ax.add_patch(Rectangle((r_e, -w / 2), 140, w, facecolor="#fdf0e6", edgecolor=ORANGE, lw=1.6,
                           zorder=3))
    ax.plot([r_e, r_e + 140], [0, 0], color=ORANGE, lw=1.2, zorder=4)
    ax.text(r_e + 70, -w / 2 - 12, "exit-lane corridor\n(flush at overlap = 0)", ha="center",
            va="top", fontsize=8.5, color=ORANGE)
    ax.text(0, -r_e - 40, "exit_radius = terminal_radius + corridor_width/2 − overlap\n"
            "overlap > 0 penetrates the column · overlap < 0 leaves a gap", ha="center", va="top",
            fontsize=9, color=INK)
    ax.set_title("exit_radius — the exit-lane inner edge relative to the hub column", fontsize=11,
                 color=INK, pad=8)
    lim = r_e + 160
    ax.set_xlim(-lim * 0.6, lim)
    ax.set_ylim(-r_e - 80, r_e + 20)
    _save(fig, "exit_radius")


def fig_segment_overlaps_column() -> None:
    """volumes.segment_overlaps_column: distance(centre, extended centreline) < R + width/2."""
    w, R = 60.0, 90.0
    ext = hw = w / 2
    cx, cy = 0.0, 0.0
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((cx, cy), R, facecolor="#edf2f7", edgecolor=INK, lw=1.6, zorder=1))
    ax.add_patch(Circle((cx, cy), R + hw, facecolor="none", edgecolor=GRID, lw=1.2, ls=":",
                        zorder=1))
    ax.plot([cx], [cy], "o", color=INK, ms=5, zorder=5)
    ax.text(R + 16, 0, "hub column\n(radius R)", ha="left", va="center", fontsize=9, color=INK)
    ax.text(cx, -(R + hw) - 10, "dotted = test radius  R + width/2", ha="center", va="top",
            fontsize=8, color=GRID)

    def draw_seg(y, color, label, dy):
        """Draw a horizontal segment at height ``y``, its extended centreline, and centre→line dist."""
        ax0, bx0 = -130.0, 250.0
        ax.plot([ax0, bx0], [y, y], color=color, lw=2.6, marker="o", ms=4, zorder=4)
        ax.plot([ax0 - ext, bx0 + ext], [y, y], color=color, lw=1.0, ls="--", zorder=3)
        ax.plot([cx, cx], [cy, y], color=color, lw=1.2, ls="-.", zorder=3)
        d = abs(y - cy)
        verdict = "TAGGED  (d < R + width/2)" if d < R + hw else "untagged  (d ≥ R + width/2)"
        ax.text(cx + 8, (cy + y) / 2, f"d = {d:.0f}", fontsize=8, color=color, ha="left",
                va="center")
        ax.text((ax0 + bx0) / 2, y + dy, f"{label} → {verdict}", fontsize=9, color=color,
                ha="center", va="center", weight="bold")

    draw_seg(105.0, ORANGE, "near-hub box", 26)
    draw_seg(-205.0, BLUE, "far cruise box", -26)
    ax.set_title("segment_overlaps_column — distance(centre, extended centreline) < R + width/2",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-175, 330)
    ax.set_ylim(-250, 165)
    _save(fig, "segment_overlaps_column")


def fig_fold_corners() -> None:
    """volumes.fold_corners_to_columns: drop in-column corners, re-root at the exit_radius edge."""
    r_e = 120.0
    center = (0.0, 0.0)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle(center, r_e, facecolor="#edf2f7", edgecolor=BLUE, lw=1.8, ls="--", zorder=1))
    ax.plot([0], [0], "o", color=INK, ms=6, zorder=6)
    ax.text(0, -20, "hub centre", ha="center", va="top", fontsize=9, color=INK)
    ax.text(-r_e * 0.62, -r_e * 0.62, "exit_radius", color=BLUE, fontsize=9, ha="right", va="top")
    raw = [(0, 0), (70, 55), (210, 120), (400, 210), (600, 250)]
    rx, ry = zip(*raw)
    ax.plot(rx, ry, color=RAW, lw=1.6, ls=":", marker="o", ms=5, zorder=3,
            label="raw path — in-column corners dropped")
    outside = [p for p in raw if (p[0] ** 2 + p[1] ** 2) ** 0.5 >= r_e]
    ep = _edge_point(center, outside[0], r_e)
    fx, fy = zip(*([ep] + outside))
    ax.plot(fx, fy, color=ORANGE, lw=2.6, marker="o", ms=4, zorder=5,
            label="folded — re-rooted at exit_radius edge")
    ax.plot([ep[0]], [ep[1]], "o", color=ORANGE, ms=8, zorder=6)
    ax.annotate("", xy=ep, xytext=center, arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.plot([], [], color=INK, lw=1.3, label="centre→edge leg — flown but UNRESERVED (column covers it)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95, borderpad=0.8)
    ax.set_title("fold_corners_to_columns — drop in-column corners, re-root at the exit_radius edge",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-190, 640)
    ax.set_ylim(-110, 300)
    _save(fig, "fold_corners")


def fig_altitude_ladder() -> None:
    """config.flight_levels_m: boxes ± height/2, gaps > corridor_height, within [ground, ceiling]."""
    ground, ceiling = 0.0, 125.0
    levels = [30.0, 70.0, 110.0]
    h = 30.0
    half = h / 2.0
    xw = 3.0
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for z, name in ((ground, "ground_level_m = 0"), (ceiling, "airspace_ceiling_m = 125")):
        ax.axhline(z, color=RED, lw=1.6)
        ax.text(xw + 0.35, z, name, va="center", fontsize=9, color=RED)
    for i, z in enumerate(levels):
        ax.add_patch(Rectangle((0, z - half), xw, h, facecolor="#ebf3fb", edgecolor=BLUE, lw=1.7))
        ax.plot([0, xw], [z, z], color=BLUE, lw=1.0, ls="--")
        ax.text(xw / 2, z, f"level {i} = {z:.0f} m", ha="center", va="center", fontsize=9, color=INK)
        ax.text(-0.25, z - half, f"{z - half:.0f}", ha="right", va="center", fontsize=7.5, color=GRID)
        ax.text(-0.25, z + half, f"{z + half:.0f}", ha="right", va="center", fontsize=7.5, color=GRID)
    ax.annotate("", xy=(xw + 0.05, levels[0] + half), xytext=(xw + 0.05, levels[0] - half),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(xw + 0.15, levels[0], "box =\nlevel ±\nheight/2", va="center", fontsize=7.5, color=INK)
    gap_lo, gap_hi = levels[0] + half, levels[1] - half
    ax.annotate("", xy=(1.3, gap_hi), xytext=(1.3, gap_lo),
                arrowprops=dict(arrowstyle="<->", color=GRID, lw=1.1))
    ax.text(1.5, (gap_lo + gap_hi) / 2,
            f"gap {levels[1] - levels[0]:.0f} m > corridor_height {h:.0f} m  ⇒  disjoint in z (FCL)",
            va="center", ha="left", fontsize=8, color=GRID)
    ax.set_title("flight_levels_m — the A* altitude ladder (single source of truth)\n"
                 "boxes ± corridor_height/2, gaps > corridor_height, within [ground, ceiling]",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-1.2, 6.6)
    ax.set_ylim(-12, 140)
    ax.axis("off")
    _save(fig, "altitude_ladder")


def fig_segment_frame() -> None:
    """geometry.segment_frame / box_from_segment: local x along segment, y lateral, oriented box."""
    p0 = np.array([1.5, 1.5])
    p1 = np.array([7.5, 4.5])
    w = 1.6
    x = (p1 - p0) / np.linalg.norm(p1 - p0)
    y = np.array([-x[1], x[0]])
    ext = w / 2
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")
    a, b = p0 - x * ext, p1 + x * ext
    corners = [a + y * w / 2, b + y * w / 2, b - y * w / 2, a - y * w / 2]
    ax.add_patch(Polygon(corners, closed=True, facecolor="#eef1f5", edgecolor=GRID, lw=1.6))
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=1.8, marker="o", ms=6, zorder=4)
    ax.text(p0[0] - 0.15, p0[1] - 0.3, "p0", fontsize=10, color=INK, ha="right")
    ax.text(p1[0] + 0.15, p1[1] + 0.2, "p1", fontsize=10, color=INK)
    m = (p0 + p1) / 2
    ax.annotate("", xy=m + x * 2.0, xytext=m, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(*(m + x * 2.1), "local x\n(along p0→p1)", color=ORANGE, fontsize=9, ha="left",
            va="center")
    ax.annotate("", xy=m + y * 1.6, xytext=m, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.text(*(m + y * 1.75), "local y\n(lateral: perp to segment & world-up\n→ level corridor is flat)",
            color=BLUE, fontsize=9, ha="center", va="bottom")
    ax.text(4.5, 0.2,
            "box_from_segment builds the oriented box on this frame (extended by width/2 each end).\n"
            "Near-vertical segment: reference falls back to world-x (avoids a degenerate cross with world-up).",
            ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_title("segment_frame — local x along the segment, y lateral (columns = local→world)",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-0.3, 10.5)
    ax.set_ylim(-1.4, 6.6)
    _save(fig, "segment_frame")


def fig_hub_placement() -> None:
    """demand: reject-sampled non-overlapping hub airspaces + Voronoi (nearest-hub) assignment."""
    w = 10.0
    hubs = [(2.2, 7.6, 0.9, "A"), (7.9, 8.1, 0.9, "A"),
            (2.6, 2.4, 0.7, "B"), (6.2, 4.4, 0.7, "B"), (8.6, 2.2, 0.7, "B")]
    pts = np.array([[x, y] for x, y, _, _ in hubs])
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_aspect("equal")
    gx, gy = np.meshgrid(np.linspace(0, w, 400), np.linspace(0, w, 400))
    flat = np.stack([gx.ravel(), gy.ravel()], axis=1)
    owner = np.argmin(np.linalg.norm(flat[:, None, :] - pts[None, :, :], axis=2), axis=1)
    tint = np.array([(0.85, 0.91, 0.97), (0.99, 0.93, 0.86)])
    is_b = np.array([op == "B" for *_, op in hubs])
    rgb = tint[is_b[owner].astype(int)].reshape(gx.shape + (3,))
    ax.imshow(rgb, origin="lower", extent=(0, w, 0, w), zorder=0, interpolation="nearest")
    for x, y, r, op in hubs:
        c = BLUE if op == "A" else ORANGE
        ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=c, lw=1.8, zorder=3))
        ax.plot([x], [y], "o", color=c, ms=6, zorder=4)
    (x0, y0), (x1, y1) = pts[3], pts[2]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, ls=":"))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.2, "≥ r_i + r_j + gap\n(reject-sampled,\nall operators)",
            ha="center", va="bottom", fontsize=8, color=INK)
    cust = np.array([4.7, 6.2])
    owner_hub = pts[int(np.argmin(np.linalg.norm(pts - cust, axis=1)))]
    ax.plot([cust[0]], [cust[1]], "*", color=INK, ms=13, zorder=5)
    ax.annotate("", xy=owner_hub, xytext=cust, arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(cust[0] + 0.15, cust[1] + 0.15, "customer →\nnearest hub", fontsize=8.5, color=INK)
    ax.plot([], [], "o", color=BLUE, label="operator A hub + keep-out")
    ax.plot([], [], "o", color=ORANGE, label="operator B hub + keep-out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("hub placement — reject-sampled non-overlapping airspaces + Voronoi cells",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(0, w)
    ax.set_ylim(0, w)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, "hub_placement")


def fig_hex_lattice_overhead() -> None:
    """viz/metrics: hex-lattice quantization overhead — off-axis flight books up to 2/√3 − 1."""
    R = 1.0
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for q in range(-1, 7):
        for r in range(-1, 6):
            cx, cy = _axial_to_xy(q, r, R)
            ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                        facecolor="white", edgecolor=HEXGRID, lw=1.0, zorder=0))
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    start, goal = (0, 0), (4, 4)
    gx, gy = _axial_to_xy(*goal, R)
    cur, path = start, [start]
    for _ in range(50):
        if cur == goal:
            break
        cur = min(((cur[0] + d[0], cur[1] + d[1]) for d in dirs),
                  key=lambda c: (_axial_to_xy(*c, R)[0] - gx) ** 2 + (_axial_to_xy(*c, R)[1] - gy) ** 2)
        path.append(cur)
    xy = [_axial_to_xy(*c, R) for c in path]
    lat = sum(((xy[i + 1][0] - xy[i][0]) ** 2 + (xy[i + 1][1] - xy[i][1]) ** 2) ** 0.5
              for i in range(len(xy) - 1))
    sx, sy = _axial_to_xy(*start, R)
    straight = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
    px, py = zip(*xy)
    ax.plot(px, py, color=ORANGE, lw=2.6, marker="o", ms=5, zorder=3,
            label=f"lattice path (hex centres) = {lat:.2f}")
    ax.plot([sx, gx], [sy, gy], color=BLUE, lw=2.4, ls="--", zorder=4,
            label=f"straight line = {straight:.2f}")
    ax.plot([sx, gx], [sy, gy], "o", color=BLUE, ms=7, zorder=5)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
    ax.text((sx + gx) / 2, min(sy, gy) - 1.9,
            f"drawn overhead = {100 * (lat / straight - 1):.1f}%     worst case  2/√3 − 1 ≈ 15.5%\n"
            "(pure hex-direction quantization — geometry, not congestion)", ha="center", va="top",
            fontsize=9.5, color=INK)
    ax.set_title("Hex-lattice quantization overhead — why detour splits into lattice vs traffic",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-2.5, 15.5)
    ax.set_ylim(-3.4, 9.6)
    _save(fig, "hex_lattice_overhead")


def fig_read_envelope() -> None:
    """parallel: padded probed-cell AABB + hub discs, and the dirty-commit intersection test."""
    R = 1.0
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_aspect("equal")
    ax.axis("off")
    probed = {(q, r) for q in range(1, 4) for r in range(1, 3)}
    xs, ys = [], []
    for q in range(-1, 6):
        for r in range(0, 4):
            cx, cy = _axial_to_xy(q, r, R)
            hit = (q, r) in probed
            ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                        facecolor="#ebf3fb" if hit else "white", edgecolor=HEXGRID,
                                        lw=1.0, zorder=1))
            if hit:
                xs.append(cx)
                ys.append(cy)
    pad = 0.9
    x0, x1 = min(xs) - R - pad, max(xs) + R + pad
    y0, y1 = min(ys) - R - pad + 0.2, max(ys) + R + pad - 0.2
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=BLUE, lw=2.0,
                           ls="--", zorder=3))
    ax.text((x0 + x1) / 2, y1 + 0.15, "read envelope = probed-cell AABB, padded by env_pad_m",
            ha="center", va="bottom", fontsize=9, color=BLUE)
    hub = (min(xs) - 2.4, max(ys) + 0.4)
    ax.add_patch(Circle(hub, 1.1, facecolor="none", edgecolor=BLUE, lw=1.6, ls="--", zorder=3))
    ax.text(hub[0], hub[1] - 1.25, "hub-read disc", ha="center", va="top", fontsize=8.5, color=BLUE)

    def commit(cx, cy, wx, wy, color, label, lx, ly):
        """Draw a committed-volume box and its dirty/clean verdict label."""
        ax.add_patch(Rectangle((cx, cy), wx, wy, facecolor=color, alpha=0.22, edgecolor=color,
                               lw=1.8, zorder=4))
        ax.text(lx, ly, label, fontsize=9, color=color, ha="center", weight="bold")

    commit(x0 + 1.0, y0 + 0.6, 1.6, 1.2, RED, "commit ∩ envelope\n→ DIRTY (replan serially)",
           x0 + 1.8, y0 - 0.55)
    commit(x1 + 1.3, y0 + 0.3, 1.4, 1.1, GREEN, "commit clear\n→ clean (keep speculation)",
           x1 + 2.0, y0 - 0.55)
    ax.set_title("parallel read envelope — a commit intersecting it makes the speculation dirty",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(min(xs) - 4.2, x1 + 3.6)
    ax.set_ylim(y0 - 1.6, y1 + 1.0)
    _save(fig, "read_envelope")


def fig_search_window() -> None:
    """astar window_bounds / planner._build_window: the reroute ellipse + padded search window."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    o, d = np.array([2.0, 2.5]), np.array([10.0, 5.0])
    for c, angs, col, name in ((o, (40, 90, 140), BLUE, "origin"), (d, (220, 270, 320), ORANGE, "dest")):
        ax.add_patch(Circle(c, 0.55, facecolor="#edf2f7", edgecolor=col, lw=1.6, zorder=3))
        ax.plot([c[0]], [c[1]], "o", color=col, ms=5, zorder=4)
        for a in angs:
            e = c + 1.2 * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
            ax.plot([c[0], e[0]], [c[1], e[1]], color=col, lw=1.2, zorder=2)
            ax.plot([e[0]], [e[1]], "o", color=col, ms=3, zorder=2)
        ax.text(c[0], c[1] - 1.0, f"{name} + lanes", ha="center", va="top", fontsize=8.5, color=col)
    lo, hi = np.minimum(o, d) - 1.3, np.maximum(o, d) + 1.3
    ax.add_patch(Rectangle(lo, *(hi - lo), facecolor="none", edgecolor=GRID, lw=1.4))
    m = 0.8
    ax.add_patch(Rectangle(lo - m, *(hi - lo + 2 * m), facecolor="none", edgecolor=INK, lw=1.4, ls="--"))
    mid = (o + d) / 2
    L = float(np.hypot(*(d - o)))
    ang = float(np.degrees(np.arctan2(d[1] - o[1], d[0] - o[0])))
    ax.add_patch(Ellipse(mid, L + 1.8, 3.2, angle=ang, facecolor=BLUE, alpha=0.12, edgecolor=BLUE, lw=1.2, zorder=1))
    ax.text(mid[0], mid[1] + 0.35, "A* reroute ellipse", fontsize=8.5, color=BLUE, ha="center")
    ax.text((lo - m)[0], (hi + m)[1] + 0.2, "search window = anchor bbox + lateral_margin ring", fontsize=9, color=INK, va="bottom")
    ax.set_title("A* search window — anchor bbox + lateral_margin must contain the reroute ellipse", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim((lo - m)[0] - 0.8, (hi + m)[0] + 0.8)
    ax.set_ylim((lo - m)[1] - 0.8, (hi + m)[1] + 0.9)
    _save(fig, "search_window")


def fig_hex_layout() -> None:
    """hexgrid: pointy-top axial (q, r) layout and the six neighbour directions."""
    R = 1.0
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for q in range(-1, 4):
        for r in range(-1, 3):
            hx, hy = _axial_to_xy(q, r, R)
            ax.add_patch(RegularPolygon((hx, hy), numVertices=6, radius=R, orientation=0, facecolor="white", edgecolor=HEXGRID, lw=1.0))
            ax.text(hx, hy, f"{q},{r}", ha="center", va="center", fontsize=7, color=GRID)
    c = (1, 1)
    ccx, ccy = _axial_to_xy(*c, R)
    ax.add_patch(RegularPolygon((ccx, ccy), numVertices=6, radius=R, orientation=0, facecolor="#ebf3fb", edgecolor=BLUE, lw=1.8, zorder=2))
    ax.text(ccx, ccy, f"{c[0]},{c[1]}", ha="center", va="center", fontsize=8, color=INK, weight="bold")
    for dq, dr in ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)):
        nx, ny = _axial_to_xy(c[0] + dq, c[1] + dr, R)
        ax.annotate("", xy=(nx, ny), xytext=(ccx, ccy), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
        ax.text((ccx + nx) / 2, (ccy + ny) / 2, f"({dq:+d},{dr:+d})", fontsize=6.5, color=ORANGE, ha="center", va="center")
    ax.text((ccx), -3.0, "x = R√3(q + r/2),   y = 1.5R·r    (pointy-top axial)", ha="center", fontsize=9, color=INK)
    ax.set_title("Hex axial (q, r) layout and the six neighbour directions", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-3.2, 8.0)
    ax.set_ylim(-3.4, 4.2)
    _save(fig, "hex_layout")


def fig_rasterisation_coverage() -> None:
    """hexgrid/occupancy: a volume dilated by the hex circumradius blocks every hex whose centre it covers."""
    R = 1.0
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    cy = _axial_to_xy(0, 2, R)[1]
    x0, x1 = _axial_to_xy(0, 2, R)[0], _axial_to_xy(4, 2, R)[0]
    hw = 0.9
    infl = hw + R
    blocked = []
    for q in range(-1, 7):
        for r in range(0, 5):
            hx, hy = _axial_to_xy(q, r, R)
            inside = (x0 - infl <= hx <= x1 + infl) and abs(hy - cy) <= infl
            ax.add_patch(RegularPolygon((hx, hy), numVertices=6, radius=R, orientation=0,
                         facecolor="#ebf3fb" if inside else "white", edgecolor=HEXGRID, lw=1.0, zorder=1))
            if inside:
                blocked.append((hx, hy))
    ax.add_patch(Rectangle((x0, cy - hw), x1 - x0, 2 * hw, facecolor="none", edgecolor=INK, lw=1.8, zorder=3))
    ax.add_patch(Rectangle((x0 - infl, cy - infl), (x1 - x0) + 2 * infl, 2 * infl, facecolor="none", edgecolor=ORANGE, lw=1.4, ls="--", zorder=3))
    for hx, hy in blocked:
        ax.plot([hx], [hy], "o", color=BLUE, ms=3, zorder=4)
    ax.text((x0 + x1) / 2, cy + infl + 0.55, "inflation halo = corridor_width/2 + circumradius R (dashed)", ha="center", fontsize=9, color=ORANGE)
    ax.text((x0 + x1) / 2, cy - infl - 0.7, "blocked = hex whose CENTRE (dot) lies in the halo — over-block by up to a hex is safe", ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_title("Conservative rasterisation — a volume dilated by R decides hex-cell membership", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(x0 - infl - 1.5, x1 + infl + 1.5)
    ax.set_ylim(cy - infl - 1.7, cy + infl + 1.5)
    _save(fig, "rasterisation_coverage")


def fig_cell_blocking() -> None:
    """astar occupancy is_blocked: own-hub flights pass the shared column, foreign cruise is walled."""
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 90, facecolor="#edf2f7", edgecolor=BLUE, lw=1.8, zorder=2))
    ax.add_patch(Circle((0, 0), 120, facecolor="none", edgecolor=GRID, lw=1.3, ls="--", zorder=2))
    ax.add_patch(Circle((0, 0), 205, facecolor="none", edgecolor=GRID, lw=1.0, ls=":", zorder=2))
    ax.plot([0], [0], "o", color=INK, ms=5, zorder=3)
    ax.text(0, -14, "hub column\n(terminal_radius 90 m)", ha="center", va="top", fontsize=8.5, color=BLUE)
    ax.text(126, 60, "exit-lane edge ~120 m", fontsize=7.5, color=GRID)
    ax.text(150, -150, "inflated keep-out\n(terminal_cells ~205 m)", fontsize=7.5, color=GRID)
    ax.annotate("", xy=(-45, 55), xytext=(-255, 195), arrowprops=dict(arrowstyle="->", color=RED, lw=2.0))
    ax.plot([-45], [55], "x", color=RED, ms=11, mew=2, zorder=5)
    ax.text(-255, 205, "foreign cruise → BLOCKED", color=RED, fontsize=9, ha="left", weight="bold")
    ax.annotate("", xy=(235, 120), xytext=(60, 25), arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
    ax.text(120, 150, "own-hub exit lane → exempt", color=GREEN, fontsize=9, ha="left", weight="bold")
    ax.text(0, -235, "but an own-only cell still carrying a committed SIBLING corridor stays blocked (fixed_exit_lanes)", ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("Cell blocking — own-hub exemption vs foreign walling", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-285, 285)
    ax.set_ylim(-265, 230)
    _save(fig, "cell_blocking")


def fig_takeoff_fan() -> None:
    """astar kernel: takeoff successors — climb straight up at the pad, then translate out to a lane × level."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 0.5, facecolor="#edf2f7", edgecolor=BLUE, lw=1.6, zorder=3))
    ax.plot([0], [0], "o", color=INK, ms=6, zorder=4)
    ax.annotate("↑ climb (z) at the pad", xy=(0, 0), xytext=(0.2, 0.5), fontsize=8, color=INK)
    ax.text(0, -0.8, "pad / column", ha="center", va="top", fontsize=8.5, color=BLUE)
    for a in range(0, 360, 60):
        e = 2.6 * np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
        ax.plot([e[0]], [e[1]], "o", color=ORANGE, ms=4)
    ax.text(0, 3.1, "then translate out to one of N exit lanes at cruise level", ha="center", fontsize=9, color=ORANGE)
    ax.text(0, -3.0, "successor order: for lane × for level", ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_title("A* takeoff fan — climb at the pad, then lane × level successors", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.6, 3.6)
    _save(fig, "takeoff_fan")


def fig_batched_turns() -> None:
    """shortcut batched_turns: seed one turn, then probe maximal straight runs on each side."""
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_aspect("equal")
    ax.axis("off")
    pts = {"A": (0, 0), "B": (1.5, 0.12), "C": (3, 0.05), "D": (4.5, 0.12), "E": (6, 0),
           "F": (6.9, 1.1), "G": (8, 2.2), "H": (9.4, 2.85), "I": (10.8, 3.4)}
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    ax.plot(xs, ys, color=GRID, lw=1.2, ls=":", marker="o", ms=4, zorder=1)
    for k, (x, y) in pts.items():
        ax.text(x, y + 0.2, k, fontsize=8, color=INK, ha="center")
    ax.plot([pts["A"][0], pts["E"][0]], [pts["A"][1], pts["E"][1]], color=BLUE, lw=2.4, zorder=2)
    ax.text(3, -0.55, "incoming run A…E", color=BLUE, fontsize=8.5, ha="center")
    ax.plot([pts["G"][0], pts["I"][0]], [pts["G"][1], pts["I"][1]], color=ORANGE, lw=2.4, zorder=2)
    ax.text(9.6, 3.0, "outgoing run G…I", color=ORANGE, fontsize=8.5, ha="left")
    ax.annotate("", xy=pts["G"], xytext=pts["E"], arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(6.5, 1.05, "seed turn E→G", color=RED, fontsize=8.5, ha="left")
    ax.text(5.4, -1.3, "probe order: seed E→G, batch A→G, fallbacks D/C/B→G, then A→I with →H fallbacks", ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("Batched turns — one seed turn, then probe maximal straight runs on each side", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-1, 12.2)
    ax.set_ylim(-1.9, 4.2)
    _save(fig, "batched_turns")


def fig_milp_obstacles() -> None:
    """MILP: reachability-lens obstacle pruning (left) and per-segment half-space keep-out by convexity (right)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    axl.set_aspect("equal")
    axl.axis("off")
    s, g = np.array([0.0, 0.0]), np.array([8.0, 0.0])
    cap, k, n = 2.2, 2, 5
    axl.add_patch(Circle(s, k * cap, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=1.2))
    axl.add_patch(Circle(g, (n - 1 - k) * cap, facecolor=ORANGE, alpha=0.10, edgecolor=ORANGE, lw=1.2))
    axl.plot([s[0]], [s[1]], "o", color=INK, ms=6)
    axl.plot([g[0]], [g[1]], "o", color=INK, ms=6)
    axl.text(s[0], s[1] - 0.6, "start", ha="center", va="top", fontsize=8.5, color=INK)
    axl.text(g[0], g[1] - 0.6, "goal", ha="center", va="top", fontsize=8.5, color=INK)
    axl.plot([2.0], [3.0], "s", color=GREEN, ms=9)
    axl.text(2.0, 3.4, "in BOTH lenses → kept", ha="center", fontsize=7.5, color=GREEN)
    axl.plot([6.8], [4.6], "x", color=RED, ms=11, mew=2)
    axl.text(6.8, 5.0, "outside both → pruned", ha="center", fontsize=7.5, color=RED)
    axl.text(4, -3.0, "segment k: disk k·cap at start, (N−1−k)·cap at goal", ha="center", fontsize=8.5, color=INK)
    axl.set_title("Reachability-lens pruning", fontsize=10, color=INK)
    axl.set_xlim(-6, 14)
    axl.set_ylim(-3.8, 6)
    axr.set_aspect("equal")
    axr.axis("off")
    axr.add_patch(Rectangle((3, -0.5), 3, 4, facecolor="#f0f0f0", edgecolor=GRID, lw=1.2))
    axr.text(4.5, 1.5, "obstacle", ha="center", fontsize=8, color=GRID)
    axr.plot([3, 3], [-2, 5], color=RED, lw=1.6, ls="--")
    axr.text(3.15, 4.7, "face half-space", color=RED, fontsize=8)
    axr.plot([0, 2], [0, 3], color=BLUE, lw=2.4, marker="o", ms=6)
    axr.text(0.2, 1.4, "segment\n(both endpoints\nbeyond the face)", fontsize=7.5, color=BLUE)
    axr.text(2.5, -3.0, "both endpoints on the outer side ⇒ by convexity the whole segment is outside", ha="center", fontsize=8, color=INK)
    axr.set_title("Per-segment half-space keep-out", fontsize=10, color=INK)
    axr.set_xlim(-2.5, 7)
    axr.set_ylim(-3.8, 5.5)
    fig.suptitle("MILP obstacle handling", fontsize=11.5, color=INK, y=1.0)
    _save(fig, "milp_obstacles")


def fig_hover_tail_steps() -> None:
    """compiled_hex_occupancy.hover_tail_steps: why the landing-column tail is ``ceil(...) + 2``.

    A committed landing column is marked by ``hexgrid._step_range`` as ``floor((t_end + dt +
    time_buffer_s)/dt)``. Two discretisation effects push the number of tail steps ABOVE the naive
    ``ceil((hover+climb+buffer)/dt)``: ``_step_range``'s own ``+dt`` widening, and the floor slip when
    the arrival lands mid-step. The ``+2`` is integer headroom that dominates both, so ``MAXS`` (which
    only sizes the box) is always a safe upper bound. Schematic, dt := one drawing cell."""
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.axis("off")

    s_a = 2                       # arrival step (integer boundary on the ruler)
    f = 0.55                      # arrival lands MID-step (fraction past the boundary)
    t_a = s_a + f
    H = 3.2                       # hover + max climb, in dt units
    b = 0.45                      # time_buffer_s, in dt units (the ASTM buffer — already counted)
    widen = 1.0                   # _step_range's explicit + dt
    t_end = t_a + H
    top = t_end + b + widen       # the continuous value _step_range floors
    s1 = int(np.floor(top))       # top blocked step
    y, hbar = 1.0, 0.46

    x0, x1 = 0, 9
    for s in range(x0, x1 + 1):                       # step boundaries
        ax.axvline(s, color=HEXGRID, lw=0.8, zorder=0)
        ax.text(s, -0.16, str(s), ha="center", va="top", fontsize=7.5, color=GRID)
    ax.plot([x0, x1], [0, 0], color=INK, lw=1.0, zorder=1)
    ax.text((x0 + x1) / 2, -0.52, "time  →   (each cell = one dt)", ha="center", va="top",
            fontsize=8.5, color=INK)

    def seg(a, c, color, hatch):
        ax.add_patch(Rectangle((a, y), c - a, hbar, facecolor=color, edgecolor=INK,
                               lw=1.0, hatch=hatch, alpha=0.85, zorder=2))
    seg(t_a, t_end, BLUE, None)
    seg(t_end, t_end + b, GREEN, "///")
    seg(t_end + b, top, ORANGE, "xxx")
    ax.text((t_a + t_end) / 2, y + hbar / 2, "hover + climb", ha="center", va="center",
            fontsize=8.5, color="white", weight="bold", zorder=3)

    # legend for the two narrow tail segments (keeps the timeline uncluttered)
    lx = 0.2
    for color, hatch, txt in ((GREEN, "///", "+ time_buffer_s  (ASTM; already counted)"),
                              (ORANGE, "xxx", "+ dt  (_step_range widening)")):
        ax.add_patch(Rectangle((lx, 2.28), 0.30, 0.22, facecolor=color, edgecolor=INK,
                               hatch=hatch, alpha=0.85))
        ax.text(lx + 0.42, 2.39, txt, ha="left", va="center", fontsize=8, color=INK)
        lx += 4.5

    ax.plot([t_a], [y], "o", color=INK, ms=6, zorder=4)                  # arrival, mid-step
    ax.plot([t_a, t_a], [0, y], ls="--", color=INK, lw=1.0, zorder=1)
    ax.text(t_a, y + hbar + 0.16, "arrival  (mid-step ⇒ floor slip)", ha="center",
            va="bottom", fontsize=8, color=INK)

    ys = y + hbar + 0.18                                                  # floor-snap, above the bar
    ax.plot([top], [ys], "o", color=ORANGE, ms=5, zorder=4)
    ax.annotate("", xy=(s1, ys), xytext=(top, ys),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(top + 0.08, ys, "⌊·/dt⌋", ha="left", va="center", fontsize=8, color=RED)
    ax.plot([s1, s1], [0, y], ls=":", color=RED, lw=1.2, zorder=1)
    ax.text(s1, 0.10, "s1", ha="center", va="bottom", fontsize=8.5, color=RED, weight="bold")

    ceil_steps = int(np.ceil(H + b))     # ceil((hover+climb+buffer)/dt)  — the numerator
    tail = s1 - s_a                      # steps actually marked past arrival
    box = ceil_steps + 2                 # hover_tail_steps

    def bracket(depth, end, color, label):
        yb = -0.98 - depth * 0.60
        ax.annotate("", xy=(end, yb), xytext=(s_a, yb),
                    arrowprops=dict(arrowstyle="|-|", color=color, lw=1.4))
        ax.text((s_a + end) / 2, yb - 0.10, label, ha="center", va="top", fontsize=8, color=color)
    bracket(0, s_a + ceil_steps, GRID, f"ceil((hover+climb+buffer)/dt) = {ceil_steps}   ← undercounts")
    bracket(1, s_a + tail, RED, f"marked tail = s1 − s_a = {tail}")
    bracket(2, s_a + box, GREEN, f"hover_tail_steps = ceil + 2 = {box}   (box depth: always ≥ tail)")

    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(-3.1, 2.65)
    _save(fig, "hover_tail_steps")


def fig_sipp_safe_intervals() -> None:
    """sipp: per-cell safe intervals (left) and safe-interval successor expansion (right)."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axl.set_title("Safe intervals for one hex cell", fontsize=10.5, color=INK)
    axl.axis("off")
    w0, w1 = 0.0, 14.0
    axl.plot([w0, w1], [2.0, 2.0], color=INK, lw=1.0, zorder=1)
    for s in range(int(w0), int(w1) + 1):
        axl.plot([s, s], [1.95, 2.05], color=GRID, lw=0.8)
    axl.text((w0 + w1) / 2, 1.5, "step →   over the window [ws0, ws1]", ha="center", va="top", fontsize=8.5, color=INK)
    for a, b in ((3, 5), (9, 11)):
        axl.add_patch(Rectangle((a, 2.12), b - a, 0.5, facecolor=RED, alpha=0.6, edgecolor=INK, lw=1.0))
    axl.text(3, 2.78, "blocked spans (committed corridor + column claims)", fontsize=8.5, color=RED, va="bottom")
    for a, b in ((0, 3), (5, 9), (11, 14)):
        axl.add_patch(Rectangle((a, 1.28), b - a, 0.45, facecolor=GREEN, alpha=0.4, edgecolor=GREEN, lw=1.2))
    axl.text(0, 0.98, "free intervals SIPP searches — the complement over the window", fontsize=8.5, color=GREEN, va="top")
    axl.set_xlim(-0.6, 14.8)
    axl.set_ylim(0.4, 3.2)

    axr.set_title("Safe-interval successor expansion", fontsize=10.5, color=INK)
    axr.axis("off")
    sx, sy = 0.0, 2.0
    axr.add_patch(Rectangle((sx - 0.7, sy - 0.35), 1.7, 0.7, facecolor="#ebf3fb", edgecolor=BLUE, lw=1.6))
    axr.text(sx + 0.15, sy, "state\n(cell, interval)", ha="center", va="center", fontsize=8, color=BLUE)
    for i, ny in enumerate((3.0, 2.0, 1.0)):
        axr.add_patch(Rectangle((3.6, ny - 0.22), 1.4, 0.44, facecolor="#f0fff4", edgecolor=GREEN, lw=1.2))
        axr.text(4.3, ny, f"nbr {i + 1} free iv", ha="center", va="center", fontsize=7, color=GREEN)
        axr.annotate("", xy=(3.6, ny), xytext=(sx + 1.0, sy), arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    axr.text(4.3, 3.6, "one successor per reachable neighbour interval\n(pre-move hover folded in)", ha="center", fontsize=7.5, color=INK)
    axr.annotate("", xy=(sx + 0.15, sy + 1.35), xytext=(sx + 0.15, sy + 0.4), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    axr.annotate("", xy=(sx + 0.15, sy - 1.35), xytext=(sx + 0.15, sy - 0.4), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    axr.text(sx + 0.35, sy + 1.4, "L+1 rung", fontsize=7.5, color=ORANGE, va="bottom")
    axr.text(sx + 0.35, sy - 1.4, "L−1 rung (both levels clear across the climb window)", fontsize=7.5, color=ORANGE, va="top")
    axr.set_xlim(-1.4, 6.4)
    axr.set_ylim(0.2, 4.0)

    fig.suptitle("SIPP — safe intervals from the claim arena, expanded interval-by-interval", fontsize=11.5, color=INK, y=1.0)
    _save(fig, "sipp_safe_intervals")


def fig_lns_anytime_loop() -> None:
    """lns.solver.run_lns + state.try_repair: destroy→repair→accept-iff-cheaper, monotone curve."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})

    axl.set_title("One iteration = one ledger transaction", fontsize=10.5, color=INK)
    axl.axis("off")
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 10)
    _box(axl, 5, 9.2, "pick operator  (adaptive weights)", fc="#ebf3fb", ec=BLUE)
    _box(axl, 5, 7.4, "DESTROY: tombstone victims\n(release_many)", fc="#fdf0e6", ec=ORANGE)
    _box(axl, 5, 5.6, "REPAIR: replan victims in order\n(shortcut / A* / SIPP)", fc="#fdf0e6", ec=ORANGE)
    _box(axl, 5, 3.8, "conflict-free  AND  cheaper?", ec=INK)
    _box(axl, 2.1, 1.7, "ADOPT\n(commit repair)", fc="#f0fff4", ec=GREEN, tc=GREEN)
    _box(axl, 7.9, 1.7, "REVERT\n(rewind: re-commit\nold volumes)", fc="#fff5f5", ec=RED, tc=RED)
    for y0, y1 in ((8.7, 8.0), (6.9, 6.2), (5.1, 4.4)):
        axl.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    axl.annotate("", xy=(2.6, 2.3), xytext=(4.2, 3.3),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    axl.text(2.6, 3.0, "yes", fontsize=8, color=GREEN, ha="center")
    axl.annotate("", xy=(7.4, 2.3), xytext=(5.8, 3.3),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axl.text(7.4, 3.0, "no", fontsize=8, color=RED, ha="center")
    axl.annotate("", xy=(5.2, 8.9), xytext=(2.1, 2.4),
                 arrowprops=dict(arrowstyle="->", color=GRID, lw=1.2,
                                 connectionstyle="arc3,rad=-0.42"))
    axl.text(0.7, 6.0, "next iteration\n(seed = new\nSeedSequence)", fontsize=7.5, color=GRID,
             ha="center", rotation=90, va="center")

    axr.set_title("Anytime incumbent — monotone, stoppable at any iteration", fontsize=10.5, color=INK)
    rng = np.random.default_rng(5)
    cost = [100.0]
    accepted = []
    for i in range(46):
        if rng.random() < 0.32:
            cost.append(cost[-1] - rng.uniform(1.0, 7.0))
            accepted.append(i + 1)
        else:
            cost.append(cost[-1])
    axr.step(range(len(cost)), cost, where="post", color=BLUE, lw=2.0, zorder=2)
    axr.plot(accepted, [cost[i] for i in accepted], "o", color=GREEN, ms=5, zorder=3,
             label="accepted (cost drops)")
    axr.plot([i for i in range(1, len(cost)) if i not in accepted],
             [cost[i] for i in range(1, len(cost)) if i not in accepted], "x", color=RAW, ms=5,
             zorder=3, label="rejected (revert; flat)")
    axr.set_xlabel("iteration", fontsize=9, color=INK)
    axr.set_ylabel("incumbent cost", fontsize=9, color=INK)
    axr.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95)
    axr.text(1, cost[-1] + 1.5, "never increases — a rejected repair leaves the ledger untouched",
             fontsize=8.5, color=INK)
    axr.grid(True, color=HEXGRID, lw=0.6, zorder=0)

    fig.suptitle("LNS anytime loop — accept only a conflict-free, strictly cheaper repair",
                 fontsize=11.5, color=INK, y=1.0)
    _save(fig, "lns_anytime_loop")


def fig_lns_destroy_operators() -> None:
    """lns.neighborhood: agent-based virtual-timeline walk vs map-based hex BFS destroy operators."""
    R = 1.0
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.0, 5.4))

    axl.set_title("agent_based — random walk on one flight, restricted to\n"
                  "moves that can still beat the incumbent arrival", fontsize=10, color=INK)
    axl.set_aspect("equal")
    axl.axis("off")
    goal = (6, 0)
    for q in range(-1, 8):
        for r in range(-2, 3):
            cx, cy = _axial_to_xy(q, r, R)
            if _hex_dist((q, r), goal) > 7:
                continue
            axl.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                         facecolor="white", edgecolor=HEXGRID, lw=0.8, zorder=0))
    walk = [(0, 0), (1, 0), (2, -1), (3, -1), (4, 0), (5, 0)]
    arrival = 8
    for k, cell in enumerate(walk):
        cx, cy = _axial_to_xy(*cell, R)
        # a move is legal only while t+1 + steps_to(goal) < arrival_step (shrinking budget cone)
        slack = arrival - (k + _hex_dist(cell, goal))
        fc = "#ebf3fb" if slack > 0 else "#fff5f5"
        axl.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                     facecolor=fc, edgecolor=BLUE, lw=1.3, zorder=1))
        axl.text(cx, cy + 0.32, f"t={k}", ha="center", fontsize=7, color=INK)
    wx = [_axial_to_xy(*c, R)[0] for c in walk]
    wy = [_axial_to_xy(*c, R)[1] for c in walk]
    axl.plot(wx, wy, color=BLUE, lw=2.2, marker="o", ms=4, zorder=2)
    sx, sy = _axial_to_xy(*walk[0], R)
    axl.annotate("seed\n(most-delayed flight)", xy=(sx, sy + 0.5), xytext=(sx - 0.4, sy + 2.2),
                 ha="center", fontsize=7.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    gx, gy = _axial_to_xy(*goal, R)
    axl.add_patch(RegularPolygon((gx, gy), numVertices=6, radius=R, orientation=0,
                                 facecolor="#edf2f7", edgecolor=INK, lw=1.4, zorder=1))
    axl.text(gx, gy, "goal", ha="center", va="center", fontsize=7.5, color=INK)
    for cell, col in (((2, -1), ORANGE), ((4, 0), GREEN)):
        cx, cy = _axial_to_xy(*cell, R)
        axl.plot([cx + 0.28], [cy - 0.28], "*", color=col, ms=12, zorder=4)
    axl.text((sx + gx) / 2, -4.0, "walk start ∈ [unimpeded launch, arrival); each move must keep "
             "t+1+dist(goal) < arrival\n★ = owner of a claim the walk hits → collected as a victim",
             ha="center", va="top", fontsize=7.5, color=INK)
    axl.set_ylim(-5.0, 4.2)

    axr.set_title("map_based — BFS outward from a random contention cell,\n"
                  "collecting claimants of each contended cell reached", fontsize=10, color=INK)
    axr.set_aspect("equal")
    axr.axis("off")
    start = (3, 1)
    contended = {(3, 1), (4, 0), (2, 2), (5, 1), (1, 2)}
    for q in range(0, 7):
        for r in range(-1, 4):
            cx, cy = _axial_to_xy(q, r, R)
            ring = _hex_dist((q, r), start)
            if ring > 3:
                continue
            fc = ("#dbeafe", "#e8f0fb", "#f2f6fc", "white")[min(ring, 3)]
            if (q, r) in contended:
                fc = "#fdf0e6"
            axr.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                         facecolor=fc, edgecolor=HEXGRID, lw=0.8, zorder=1))
            if (q, r) in contended:
                axr.plot([cx], [cy], "*", color=ORANGE, ms=13, zorder=3)
    scx, scy = _axial_to_xy(*start, R)
    axr.add_patch(RegularPolygon((scx, scy), numVertices=6, radius=R, orientation=0,
                                 facecolor="none", edgecolor=RED, lw=2.2, zorder=2))
    axr.text(scx, scy - 1.5, "start = random\ncontention cell", ha="center", va="top", fontsize=7.5,
             color=RED)
    for ring, lbl in ((1, "BFS ring 1"), (2, "ring 2"), (3, "ring 3")):
        ex, ey = _axial_to_xy(start[0] + ring, start[1], R)
        axr.text(ex, ey + 0.75, lbl, ha="center", fontsize=7, color=BLUE)
    axr.text(scx, scy + 2.55, "★ contended cell → outward-in-time sweep (δ = 0, ±1, …)\n"
             "collects every movable claimant", ha="center", fontsize=7.5, color=INK)
    axr.set_ylim(-2.8, 4.6)

    fig.suptitle("LNS destroy operators — how each grows a neighborhood to re-plan",
                 fontsize=11.5, color=INK, y=1.0)
    _save(fig, "lns_destroy_operators")


def fig_lns_drop_vs_sync() -> None:
    """lns.parallel: SYNC per-round barrier (idle lanes) vs DROP apply-on-arrival (no idle)."""
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(11.2, 6.2), sharex=True)
    m = 3
    lanes = [2.5, 1.5, 0.5]

    def task(ax, lane, x0, dur, color, edge=INK):
        ax.add_patch(Rectangle((x0, lane - 0.28), dur, 0.56, facecolor=color, alpha=0.75,
                               edgecolor=edge, lw=1.2, zorder=2))

    axt.set_title("SYNC — barrier per round: dispatch m, wait all, apply best-of-m, discard m−1",
                  fontsize=10.5, color=INK)
    rounds = [(0.0, [1.4, 2.6, 1.9]), (2.9, [2.2, 1.6, 2.0]), (5.2, [1.7, 2.3, 1.5])]
    for r0, durs in rounds:
        barrier = r0 + max(durs) + 0.1
        best = int(np.argmax(durs)) if False else int(np.argmin(durs))  # pick one "winner"
        best = 1
        for j, dur in enumerate(durs):
            task(axt, lanes[j], r0, dur, BLUE if j == best else RAW)
            if r0 + dur < barrier:  # idle wait until the barrier
                axt.add_patch(Rectangle((r0 + dur, lanes[j] - 0.28), barrier - (r0 + dur), 0.56,
                                        facecolor="none", edgecolor=GRID, lw=0.9, ls=":", zorder=1,
                                        hatch="///"))
            if j == best:
                axt.plot([r0 + dur], [lanes[j]], "*", color=GREEN, ms=13, zorder=4)
            else:
                axt.text(r0 + dur - 0.2, lanes[j], "✕", color=RED, ha="center", va="center",
                         fontsize=9, zorder=4)
        axt.axvline(barrier, color=INK, lw=1.4, ls="--", zorder=3)
    axt.text(0.05, 3.35, "hatched = worker idle at the barrier   ★ applied best   ✕ discarded",
             fontsize=8, color=INK)
    axt.set_ylim(0, 3.6)
    axt.set_yticks(lanes)
    axt.set_yticklabels([f"w{w}" for w in range(m)], fontsize=8)

    axb.set_title("DROP — no barrier: apply on arrival, re-dispatch immediately (no lane idles)",
                  fontsize=10.5, color=INK)
    schedule = [(0, 0.0, 1.4, "clean"), (1, 0.0, 2.6, "overwrite"), (2, 0.0, 1.9, "clean"),
                (0, 1.4, 1.8, "discard"), (2, 1.9, 1.5, "clean"), (1, 2.6, 1.7, "clean"),
                (0, 3.2, 2.1, "clean"), (2, 3.4, 1.9, "overwrite"), (1, 4.3, 1.6, "clean")]
    verdict_col = {"clean": GREEN, "overwrite": ORANGE, "discard": RED}
    for w, x0, dur, verdict in schedule:
        task(axb, lanes[w], x0, dur, "#cbd5e0")
        axb.plot([x0 + dur], [lanes[w]], "o", color=verdict_col[verdict], ms=6, zorder=4)
    axb.text(0.05, 3.35, "result arrives against a STALE base_version → "
             "clean (merge) · overwrite (whole soln still wins) · discard",
             fontsize=8, color=INK)
    axb.set_ylim(0, 3.6)
    axb.set_yticks(lanes)
    axb.set_yticklabels([f"w{w}" for w in range(m)], fontsize=8)
    axb.set_xlabel("wall clock  →", fontsize=9, color=INK)
    axb.set_xlim(-0.15, 8.2)

    fig.suptitle("Parallel LNS scheduling — SYNC buys wall clock, DROP converts it to accepts",
                 fontsize=11.5, color=INK, y=0.99)
    _save(fig, "lns_drop_vs_sync")


def fig_od_hop_ellipse() -> None:
    """colgen network/dp_kernel/pricing: the O-D hop-ellipse corridor.

    A priced route stays within ``shortest + overrun`` summed hops, so the reachable cells are
    exactly ``hex_dist(o, c) + hex_dist(c, d) <= shortest + overrun`` — the hop budget IS the
    corridor radius. ``overrun`` = ``params.max_air_overrun_hops`` (0 pins the geodesic band).
    """
    R = 1.0
    origin, dest = (0, 0), (6, 0)
    overrun = 2
    shortest = _hex_dist(origin, dest)
    radius = shortest + overrun
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for q in range(-3, 10):
        for r in range(-5, 6):
            if _hex_dist(origin, (q, r)) > radius:
                continue
            cx, cy = _axial_to_xy(q, r, R)
            d = _hex_dist(origin, (q, r)) + _hex_dist((q, r), dest)
            if d <= shortest:
                fc = "#fdf0e6"          # geodesic band (d == shortest)
            elif d <= radius:
                fc = "#ebf3fb"          # inside the overrun corridor
            else:
                fc = "white"
            ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                        facecolor=fc, edgecolor=HEXGRID, lw=0.8, zorder=1))
    for cell, name, col in ((origin, "O", BLUE), (dest, "D", ORANGE)):
        cx, cy = _axial_to_xy(*cell, R)
        ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                    facecolor=col, edgecolor=INK, lw=1.6, zorder=3))
        ax.text(cx, cy, name, ha="center", va="center", fontsize=12, color="white", weight="bold",
                zorder=4)
    ax.plot([], [], "s", color="#fdf0e6", markeredgecolor=HEXGRID,
            label="geodesic cells  (d = shortest)")
    ax.plot([], [], "s", color="#ebf3fb", markeredgecolor=HEXGRID,
            label=f"overrun corridor  (shortest < d ≤ shortest+{overrun})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("O-D hop-ellipse — priced cells satisfy hex_dist(O,c) + hex_dist(c,D) ≤ "
                 "shortest + overrun\n(the hop budget IS the corridor radius; foreign keep-out is "
                 "removed separately)", fontsize=10.5, color=INK, pad=8)
    _save(fig, "od_hop_ellipse")


def fig_pricing_dag() -> None:
    """colgen dp_kernel._price_dag / network: layered cell×step space-time DAG, parity-swapped tables."""
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.axis("off")
    n_steps, cells = 6, [3.0, 2.0, 1.0, 0.0]
    dx = 1.9
    # parity bands: even steps use table A, odd steps table B (swapped once per step)
    for s in range(n_steps):
        band = "#ebf3fb" if s % 2 == 0 else "#fdf0e6"
        ax.add_patch(Rectangle((s * dx - 0.55, -0.7), 1.1, 4.4, facecolor=band, alpha=0.5,
                               edgecolor="none", zorder=0))
        ax.text(s * dx, 3.95, f"step {s}\ntbl {'A' if s % 2 == 0 else 'B'}", ha="center",
                fontsize=7.5, color=INK)
    reach = {0: [1], 1: [1, 2], 2: [1, 2], 3: [1, 2], 4: [1, 2], 5: [1]}
    for s, rows in reach.items():
        for c in rows:
            x, y = s * dx, cells[c]
            ax.add_patch(Circle((x, y), 0.16, facecolor="white", edgecolor=INK, lw=1.2, zorder=3))
    def arc(s0, c0, c1, color, lw=1.2, z=2):
        ax.annotate("", xy=(s0 * dx + dx, cells[c1]), xytext=(s0 * dx, cells[c0]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=6, shrinkB=6),
                    zorder=z)
    # first arcs (origin/root → layer 1) in red; internal in grey; last arcs (→ dest) in green
    arc(0, 1, 1, RED, 1.8)
    arc(0, 1, 2, RED, 1.8)
    for s, (c0, c1) in ((1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (1, (2, 2)),
                        (2, (1, 1)), (2, (2, 2)), (3, (1, 1)), (3, (2, 2))):
        arc(s, c0, c1, GRID)
    arc(4, 1, 1, GREEN, 1.8)
    arc(4, 2, 1, GREEN, 1.8)
    ax.plot([0], [cells[1]], "o", color=RED, ms=9, zorder=4)
    ax.text(-0.35, cells[1], "root\n(origin,\nseeded at\nstep start)", ha="right", va="center",
            fontsize=7.5, color=RED)
    dxp, dyp = 5 * dx, cells[1]
    ax.add_patch(RegularPolygon((dxp, dyp), numVertices=6, radius=0.22, orientation=0,
                                facecolor=GREEN, edgecolor=INK, lw=1.2, zorder=4))
    ax.text(dxp + 0.35, dyp, "sink\n(dest)", ha="left", va="center", fontsize=7.5, color=GREEN)
    ax.plot([], [], color=RED, lw=1.8, label="first arc (origin role)")
    ax.plot([], [], color=GRID, lw=1.2, label="internal arc")
    ax.plot([], [], color=GREEN, lw=1.8, label="last arc (dest role)")
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    ax.set_title("Pricing DAG — cell×step nodes, hop arcs by role; two dominance tables swap by\n"
                 "step parity (never compare labels at different steps); roots seeded before arcs "
                 "write s+1", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-1.8, 11.0)
    ax.set_ylim(-0.9, 4.4)
    _save(fig, "pricing_dag")


def fig_label_dp_dominance() -> None:
    """colgen dp_kernel: coexisting labels per (cell, hop) keyed by the `recent` tail; window = depth."""
    R = 1.0
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

    axl.set_title("`recent` = last min(hop+1, depth) cells of the path\n"
                  "(depth = revisit window width − 1)", fontsize=10, color=INK)
    axl.set_aspect("equal")
    axl.axis("off")
    for q in range(-1, 8):
        for r in range(-2, 3):
            cx, cy = _axial_to_xy(q, r, R)
            axl.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                         facecolor="white", edgecolor=HEXGRID, lw=0.7, zorder=0))
    depth = 3
    path = [(0, 0), (1, 0), (2, -1), (3, 0), (4, 1), (5, 1), (6, 1)]
    xs = [_axial_to_xy(*c, R)[0] for c in path]
    ys = [_axial_to_xy(*c, R)[1] for c in path]
    axl.plot(xs, ys, color=BLUE, lw=2.0, marker="o", ms=4, zorder=2)
    for c in path[-depth:]:              # the recent tail: last `depth` cells, part of the key
        cx, cy = _axial_to_xy(*c, R)
        axl.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=0,
                                     facecolor="#ebf3fb", edgecolor=BLUE, lw=1.8, zorder=1))
    tx, ty = _axial_to_xy(*path[-1], R)
    axl.text(tx, ty + 0.85, "(cell c, hop k)", ha="center", fontsize=8, color=INK)
    hx, hy = _axial_to_xy(*path[-2], R)
    axl.text((hx + tx) / 2, ty - 1.7, f"shaded = recent tail (last {depth} cells)\n"
             "→ this label's dominance key", ha="center", va="top", fontsize=8, color=BLUE)
    ox, oy = _axial_to_xy(*path[0], R)
    axl.text(ox, oy - 1.4, "origin", ha="center", va="top", fontsize=7.5, color=INK)

    axr.set_title("Dominance key splits one (cell, hop) into many labels", fontsize=10, color=INK)
    axr.axis("off")
    axr.set_xlim(0, 10)
    axr.set_ylim(0, 10)
    _box(axr, 5, 8.8, "key = (cell, recent[0:depth], origin_paid_rows, first_hop)",
         fc="#edf2f7", fs=8)
    chips = [("recent=(c, e0, d0)", BLUE, 7.0), ("recent=(c, e1, d1)", ORANGE, 5.6),
             ("recent=(c, e2, d2)", GREEN, 4.2)]
    for txt, col, y in chips:
        _box(axr, 5, y, txt + "  first_hop=…", ec=col, tc=col, fs=8)
    axr.annotate("", xy=(5, 7.6), xytext=(5, 8.35), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axr.text(5, 2.9, "distinct keys ⇒ none dominates the others\n"
             "≈ 34 labels per (cell, hop)  (LABEL_MULTIPLICITY)\n"
             "wider revisit window ⇒ deeper key ⇒ more labels",
             ha="center", va="top", fontsize=8.5, color=INK)

    fig.suptitle("Label-DP dominance — window width IS the dominance-key depth",
                 fontsize=11.5, color=INK, y=1.0)
    _save(fig, "label_dp_dominance")


def fig_completion_envelope() -> None:
    """colgen pricing.completion_envelope / can_compete: keep a label if SOME hop count could win."""
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    hops = np.arange(1, 22)
    max_air_hops = 16
    # optimistic best reduced cost achievable at each hop count:
    #   benefit - pi_f - delay_lb(hops) + max_negative_credit   (delay_lb rises with hops)
    delay_lb = 0.7 * (hops - 1)
    optimistic = 9.0 - delay_lb + 2.0 * np.exp(-0.25 * (hops - 6) ** 2)
    incumbent = 3.0
    inside = hops <= max_air_hops
    ax.axhline(incumbent, color=RED, lw=1.6, ls="--", zorder=2)
    ax.text(21, incumbent + 0.15, "incumbent RC", ha="right", va="bottom", fontsize=8.5, color=RED)
    ax.bar(hops[inside], optimistic[inside], width=0.7, color="#cbd5e0", edgecolor=INK, lw=0.6,
           zorder=1)
    can = inside & (optimistic >= incumbent)
    ax.bar(hops[can], optimistic[can], width=0.7, color=GREEN, alpha=0.75, edgecolor=INK, lw=0.6,
           zorder=3, label="can compete (kept)")
    ax.bar(hops[~inside], optimistic[~inside], width=0.7, color="white", edgecolor=GRID, lw=0.6,
           ls=":", zorder=1)
    ax.axvline(max_air_hops + 0.5, color=ORANGE, lw=1.6, zorder=4)
    ax.text(max_air_hops + 0.6, optimistic.max() * 0.92, "cap at max_air_hops\n"
            "(entries past the ceiling\ndescribe completions the\nsearch cannot make)",
            ha="left", va="top", fontsize=8, color=ORANGE)
    ax.set_xlabel("total hops of the completion", fontsize=9, color=INK)
    ax.set_ylabel("optimistic reduced cost", fontsize=9, color=INK)
    ax.set_xticks(hops[::2])
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.set_title("Completion envelope — a label stays alive as soon as ANY hop count in range\n"
                 "could beat the incumbent; the whole label is pruned only when none can",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_ylim(0, optimistic.max() + 1.5)
    _save(fig, "completion_envelope")


def fig_cg_loop() -> None:
    """colgen solver.solve: seed/ladder/bootstrap init, then master LP → price → add columns loop."""
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)

    ax.text(2.0, 9.4, "INITIAL POOL", ha="center", fontsize=9, color=INK, weight="bold")
    _box(ax, 2.0, 8.3, "geodesic seed per flight", fc="#ebf3fb", ec=BLUE, fs=8)
    _box(ax, 2.0, 6.9, "departure ladder\n(pure clock shifts —\narithmetic, no DP)", fc="#ebf3fb",
         ec=BLUE, fs=8)
    _box(ax, 2.0, 5.2, "bootstrap variants\n(rank roots by score)", fc="#ebf3fb", ec=BLUE, fs=8)
    for y0, y1 in ((7.9, 7.5), (6.2, 5.9)):
        ax.annotate("", xy=(2.0, y1), xytext=(2.0, y0), arrowprops=dict(arrowstyle="->", color=BLUE,
                    lw=1.3))
    ax.annotate("", xy=(5.5, 6.1), xytext=(3.5, 5.2), arrowprops=dict(arrowstyle="->", color=INK,
                lw=1.4))

    _box(ax, 6.6, 6.3, "master LP\n→ duals π", fc="#fdf0e6", ec=ORANGE)
    _box(ax, 10.0, 6.3, "price sweep over\nper-flight DAGs\n(‖ workers)", fc="#fdf0e6", ec=ORANGE)
    _box(ax, 10.0, 3.3, "add columns with\nreduced cost < 0", fc="#fdf0e6", ec=ORANGE)
    _box(ax, 6.6, 3.3, "update UB & gap", fc="#fdf0e6", ec=ORANGE)
    ax.annotate("", xy=(8.7, 6.3), xytext=(8.0, 6.3), arrowprops=dict(arrowstyle="->", color=INK,
                lw=1.4))
    ax.annotate("", xy=(10.0, 4.0), xytext=(10.0, 5.7), arrowprops=dict(arrowstyle="->", color=INK,
                lw=1.4))
    ax.annotate("", xy=(8.0, 3.3), xytext=(9.0, 3.3), arrowprops=dict(arrowstyle="->", color=INK,
                lw=1.4))
    ax.annotate("", xy=(6.6, 5.7), xytext=(6.6, 4.0), arrowprops=dict(arrowstyle="->", color=INK,
                lw=1.4))
    ax.text(5.9, 4.85, "gap > tol\n& time left", fontsize=7.5, color=INK, ha="right")

    _box(ax, 12.6, 3.3, "final restricted-\nmaster IP + repair\n(ip_reserve_s tail)", fc="#f0fff4",
         ec=GREEN, tc=GREEN)
    ax.annotate("", xy=(11.4, 3.3), xytext=(11.0, 3.3), arrowprops=dict(arrowstyle="->",
                color=GREEN, lw=1.4))
    ax.text(11.2, 4.15, "converged /\nbound / timeout", fontsize=7.5, color=GREEN, ha="center")

    ax.text(7.0, 0.9, "A finished sweep is answer-identical serial or parallel; a sweep that hits "
            "pricing_deadline is not\n(a pool gets further through pricing_order — more pricing in "
            "the same budget, not the same answer)", ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("Column-generation loop — build a warm pool, then price against LP duals until the "
                 "gap closes", fontsize=10.5, color=INK, pad=8)
    _save(fig, "cg_loop")


def fig_pricing_pool_schedule() -> None:
    """colgen pricing_pool: longest-completed-prefix (index order) and the serial W(W-1)/2 launch ramp."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})

    axl.set_title("Accepted set = longest completed PREFIX of pricing_order\n"
                  "(keep the contiguous prefix; discard everything at/after the first timeout)",
                  fontsize=9.5, color=INK)
    deadline = 6.0
    n = 9
    finish = [1.2, 2.0, 2.7, 3.1, 6.8, 4.0, 4.5, 7.4, 5.2]   # completion time by index
    first_gap = next(i for i, f in enumerate(finish) if f > deadline)
    for i, f in enumerate(finish):
        kept = i < first_gap
        color = GREEN if kept else RED
        axl.barh(n - 1 - i, min(f, deadline + 0.9), height=0.62,
                 color=color, alpha=0.35 if not kept else 0.7, edgecolor=INK, lw=0.8, zorder=2)
        axl.plot([f], [n - 1 - i], "o", color=color, ms=6, zorder=3)
        tag = "kept" if kept else ("first timeout" if i == first_gap else "discarded (after gap)")
        axl.text(0.1, n - 1 - i, f"flight {i}", va="center", ha="left", fontsize=7.5, color=INK,
                 zorder=4)
        axl.text(min(f, deadline + 0.9) + 0.15, n - 1 - i, tag, va="center", fontsize=7,
                 color=color)
    axl.axvline(deadline, color=INK, lw=1.6, ls="--", zorder=1)
    axl.text(deadline, n + 0.1, "pricing_deadline\n(one absolute wall clock)", ha="center",
             va="bottom", fontsize=8, color=INK)
    axl.text(0.1, -0.9, "flight 5 finished early but sits AFTER the first gap → discarded "
             "(never a set with holes)", fontsize=7.5, color=RED)
    axl.set_ylim(-1.4, n + 1.2)
    axl.set_xlim(0, deadline + 2.6)
    axl.set_yticks([])
    axl.set_xlabel("wall clock  →", fontsize=8.5, color=INK)

    axr.set_title("Worker launch is parent-SERIAL:\nidle worker-seconds grow as W(W−1)/2",
                  fontsize=9.5, color=INK)
    W = 5
    spawn = 1.0
    for w in range(W):
        axr.add_patch(Rectangle((0, W - 1 - w - 0.3), w * spawn, 0.6, facecolor="none",
                                edgecolor=GRID, lw=1.0, ls=":", hatch="///", zorder=1))
        axr.add_patch(Rectangle((w * spawn, W - 1 - w - 0.3), 3.0, 0.6, facecolor=BLUE, alpha=0.6,
                                edgecolor=INK, lw=1.0, zorder=2))
        axr.text(-0.15, W - 1 - w, f"w{w}", ha="right", va="center", fontsize=8, color=INK)
    axr.text(W * spawn * 0.5, -0.7, "hatched = waiting for earlier spawns\n"
             "Σ idle = 0+1+…+(W−1) = W(W−1)/2 spawn-costs", ha="center", va="top", fontsize=8,
             color=INK)
    axr.plot([], [], color=BLUE, lw=6, alpha=0.6, label="pricing")
    axr.set_xlim(-0.6, W * spawn + 3.4)
    axr.set_ylim(-1.6, W)
    axr.set_yticks([])
    axr.set_xlabel("wall clock  →", fontsize=8.5, color=INK)

    fig.suptitle("Parallel pricing pool — determinism (index-order prefix) and the serial-launch "
                 "cost", fontsize=11.0, color=INK, y=1.01)
    _save(fig, "pricing_pool_schedule")


def _certificate_example():
    """Build the certificate-reuse example and assert cached/cold claim parity."""
    cfg = SimConfig(
        dt_s=4.0,
        nominal_speed_mps=30.0,
        flight_levels_m=(100.0,),
        horizon_s=7200.0,
        region_size_m=(60_000.0, 30_000.0),
        terminal_airspace_always_active=True,
    )
    radius = hg.circumradius(cfg)
    path = ((-3, 0), (-2, 0), (-1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1))

    def point(cell):
        """Return the ground-level world coordinate for one axial cell."""
        return np.r_[hg.hex_center(*cell, radius), cfg.ground_level_m]

    request = FlightRequest(9001, point(path[0]), point(path[-1]), 0.0)
    terms = [(point((0, 3)), Terminal("permanent", 1, radius=90.0))]
    params = ColGenParams()
    graph = build_flight_graph(request, cfg, terms, params)
    base = translate.Column(request.flight_id, 2, 0, None, None, path, 0.0)
    intent = translate.column_to_intent(base, request, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    model = cost_model(cfg, params)
    base = replace(base, delay_s=model.intent_cost(intent, cfg))
    claims = column_claims(base, graph, cfg)
    shifted = replace(
        base,
        departure_step=5,
        delay_s=base.delay_s + model.ground_weight * 3 * cfg.dt_s,
    )
    shifted_claims = column_claims(shifted, graph, cfg)
    delta = shifted.departure_step - base.departure_step
    expected = frozenset(
        RowKey.cell(*row.cell_coord, row.level, row.step + delta)
        if row.kind == "cell"
        else RowKey.term(row.terminal_id, row.step + delta)
        for row in claims
    )
    cold = column_claims(shifted, build_flight_graph(request, cfg, terms, params), cfg)
    assert shifted_claims == expected == cold
    shifted_intent = translate.column_to_intent(shifted, request, cfg)
    assert shifted_intent.status is IntentStatus.ACCEPTED
    assert abs(model.intent_cost(shifted_intent, cfg) - shifted.delay_s) < 1e-8
    assert len(intent.volumes) == len(shifted_intent.volumes)
    for before, after in zip(intent.volumes, shifted_intent.volumes, strict=True):
        assert before.shape == after.shape
        assert abs(after.t_start - before.t_start - delta * cfg.dt_s) < 1e-8
        assert abs(after.t_end - before.t_end - delta * cfg.dt_s) < 1e-8
    arrival = base.departure_step + graph.takeoff_steps[0] + len(path) - 1
    windows = _column_endpoint_steps(base, graph, cfg, arrival)
    selected = path[2:5]
    rows = {
        name: {
            str(cell): sorted(
                row.step for row in collection
                if row.kind == "cell" and row.cell_coord == cell and row.level == 0
            )
            for cell in selected
        }
        for name, collection in (("base", claims), ("shifted", shifted_claims))
    }
    assert rows["base"][str((0, 0))] == [8, 9, 10, 11]
    assert rows["shifted"][str((0, 0))] == [11, 12, 13, 14]
    data = {
        "claim_count": len(claims),
        "shown_rows": rows,
        "cell_window_offsets": derive_cell_window(cfg),
        "relative_endpoint_windows": [
            (window.start - base.departure_step, window.stop - base.departure_step)
            for window in windows
        ],
    }
    assert data["claim_count"] == 201
    return cfg, base, intent, terms, data


def _certificate_box_polygon(spec: BoxSpec) -> np.ndarray:
    """Project a filed ledger box onto the horizontal plane."""
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    local = corners * np.asarray(spec.extents) / 2
    return (local @ spec.rotation().T + np.asarray(spec.center))[:, :2]


def _certificate_panel(ax, xy, width, height, title, lines, color) -> None:
    """Draw one certificate cache/validation panel in normalized coordinates."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015",
                               facecolor=color, edgecolor="#D3DFE5", linewidth=1))
    ax.text(x + .025, y + height - .055, title, va="top", weight="bold", fontsize=15,
            color="#172F43")
    ax.text(x + .025, y + height - .118, lines, va="top", fontsize=12.5,
            linespacing=1.65, color="#172F43")


def fig_colgen_certificate_reuse() -> None:
    """Render spatial and temporal views of cached column-claim reuse with cold parity checks."""
    cfg, base, intent, terms, data = _certificate_example()
    gray, cert_ink = "#617281", "#172F43"
    blue, orange = "#1767A6", "#CE6B22"

    fig = plt.figure(figsize=(15, 7.7), facecolor="white")
    fig.suptitle("Same spatial route: reuse the validation result", fontsize=22,
                 weight="bold", color=cert_ink, y=.975)
    fig.text(.5, .918, "Departure 8 s → 20 s changes the clock; the flight's spatial shape is unchanged",
             ha="center", fontsize=13.5, color=gray)
    ax = fig.add_axes((.05, .24, .47, .60))
    radius = hg.circumradius(cfg)
    angles = np.deg2rad(30 + np.arange(6) * 60)
    for q in range(-5, 5):
        for r in range(-2, 5):
            center = hg.hex_center(q, r, radius)
            vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            ax.add_patch(Polygon(vertices, facecolor="white", edgecolor="#E1E7EC", lw=.8))
    for volume in intent.volumes:
        if isinstance(volume.shape, BoxSpec):
            ax.add_patch(Polygon(_certificate_box_polygon(volume.shape), facecolor=blue,
                                 edgecolor=blue, alpha=.19, lw=1.4))
    for center, terminal in terms:
        ax.add_patch(Circle(center[:2], terminal.radius, facecolor="#F9DDDB",
                            edgecolor="#AD514E", linestyle="--", lw=2))
        ax.text(center[0], center[1], "Permanent\nairspace", ha="center", va="center",
                fontsize=12, color="#85403F")
    xy = np.array([hg.hex_center(*cell, radius) for cell in base.cell_path])
    ax.plot(*xy.T, color=blue, lw=3, zorder=4)
    ax.scatter(*xy.T, color=blue, s=28, zorder=5)
    for name, cell in zip("ABC", base.cell_path[2:5], strict=True):
        point = hg.hex_center(*cell, radius)
        ax.annotate(name, point, xytext=(0, -28), textcoords="offset points", ha="center",
                    weight="bold", fontsize=14, color=cert_ink)
    for label, point in (("Origin", xy[0]), ("Destination", xy[-1])):
        ax.add_patch(Circle(point, cfg.effective_hover_radius_m, facecolor=blue, alpha=.15))
        ax.annotate(label, point, xytext=(0, -43), textcoords="offset points",
                    ha="center", fontsize=12, color=cert_ink)
    ax.set(xlim=(-470, 550), ylim=(-170, 475), aspect="equal")
    ax.set_axis_off()
    ax.text(.5, -.10, "Shaded polygons are actual filed corridor boxes.\n"
            "The same path, altitude, and lane choices identify the cached route.",
            ha="center", transform=ax.transAxes, fontsize=12, color=gray, linespacing=1.5)
    right = fig.add_axes((.56, .13, .40, .74))
    right.set(xlim=(0, 1), ylim=(0, 1))
    right.set_axis_off()
    _certificate_panel(right, (.025, .69), .94, .28, "First departure: validate once",
                       "Check cells, hops, endpoint / lane agreement,\n"
                       "translated geometry, detour limits, static walls.\n"
                       f"Build the union of {data['claim_count']} capacity-row IDs.", "#EAF2F9")
    _certificate_panel(right, (.025, .36), .94, .25, "Save a successful cache entry",
                       r"$K=(f, R, \ell, L_o, L_d)$" + "  (within this graph / configuration)\n"
                       r"$K\ \mapsto\ (d_0, A_0, \omega_o, \omega_d)$" + "\n"
                       "Base departure, row IDs, relative endpoint windows.", "#E9F4EF")
    _certificate_panel(right, (.025, .025), .94, .25, "Later departure: reuse + recheck",
                       "Match K; check new departure / arrival bounds.\n"
                       "Check the endpoint-window rounding guard.\n"
                       "Shift row timestamps; retain the spatial validation.", "#FFF1E5")
    for top, bottom in ((.69, .61), (.36, .275)):
        right.annotate("", (.50, bottom + .01), (.50, top - .015),
                       arrowprops=dict(arrowstyle="-|>", color=gray, lw=1.7))
    fig.text(.5, .025, "The master/IP still checks capacity conflicts with other flights for each timed alternative.",
             ha="center", color=cert_ink, fontsize=13, weight="bold")
    fig.savefig(f"{OUT}/colgen_certificate_spatial.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    fig.subplots_adjust(left=.10, right=.97, bottom=.21, top=.77, hspace=.65)
    fig.suptitle("Reuse the claim pattern by shifting its time indices", fontsize=22,
                 color=cert_ink, weight="bold", y=.97)
    fig.text(.5, .898, r"$A_{d+\Delta}=T_\Delta(A_d)$" + "     •     Δ = 3 steps = 12 seconds",
             ha="center", fontsize=17, color=cert_ink)
    fig.text(.5, .84, "One filled square = one capacity coefficient equal to 1; blank = 0",
             ha="center", fontsize=12.5, color=gray)
    cells = ((-1, 0), (0, 0), (0, 1))
    for ax, key, color, title in zip(
        axes, ("base", "shifted"), (blue, orange),
        ("Original: departure step 2 (8 s)", "Alternative: departure step 5 (20 s)"), strict=True,
    ):
        for j, cell in enumerate(cells):
            for step in range(6, 17):
                claimed = step in data["shown_rows"][key][str(cell)]
                ax.add_patch(Rectangle((step - .5, j - .35), 1, .7,
                                       facecolor=color if claimed else "#F4F7F9",
                                       edgecolor="white", lw=2))
                if claimed:
                    ax.text(step, j, "1", ha="center", va="center", color="white", fontsize=12)
        ax.set_yticks(range(3), ("Cell A", "Cell B", "Cell C"))
        ax.set(ylim=(2.65, -.7), xlim=(5.5, 16.5))
        ax.set_title(title, fontsize=14, loc="left", color=color, pad=12, weight="bold")
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=12)
        ax.set_xticks(range(6, 17))
        ax.tick_params(axis="x", labelbottom=True)
    axes[1].set_xlabel(r"Capacity-row time index $\tau$ (one step = 4 s)", fontsize=13, labelpad=10)
    fig.text(.5, .105, r"Cell B: $\{8,9,10,11\}\ \longrightarrow\ \{11,12,13,14\}$",
             ha="center", fontsize=17, color=cert_ink)
    fig.text(.5, .035, "Shown: three interior cells. The verified full row set also includes endpoint occupancy.\n"
             "If endpoint rounding changes the relative windows, rows are rebuilt while spatial validation is reused.",
             ha="center", fontsize=12, color=gray, linespacing=1.5)
    fig.savefig(f"{OUT}/colgen_certificate_temporal.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/colgen_certificate_spatial.png")
    print("wrote", f"{OUT}/colgen_certificate_temporal.png")


FIGURES = (
    fig_enroute_rulers, fig_corridor_box_extension, fig_exit_radius, fig_segment_overlaps_column,
    fig_fold_corners, fig_altitude_ladder, fig_segment_frame, fig_hub_placement,
    fig_hex_lattice_overhead, fig_read_envelope,
    fig_search_window, fig_hex_layout, fig_rasterisation_coverage, fig_cell_blocking,
    fig_takeoff_fan, fig_batched_turns, fig_milp_obstacles, fig_hover_tail_steps,
    fig_sipp_safe_intervals,
    fig_lns_anytime_loop, fig_lns_destroy_operators, fig_lns_drop_vs_sync,
    fig_od_hop_ellipse, fig_pricing_dag, fig_label_dp_dominance, fig_completion_envelope,
    fig_cg_loop, fig_pricing_pool_schedule, fig_colgen_certificate_reuse,
)


def main() -> None:
    """Render every figure in :data:`FIGURES` to ``context/figures/``."""
    for fig_fn in FIGURES:
        fig_fn()


if __name__ == "__main__":
    main()
