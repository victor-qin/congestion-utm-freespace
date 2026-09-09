"""Generate every ``context/figures/*.png`` schematic cited from ``freespace_sim`` docstrings.

Each ``fig_*`` renders one PNG illustrating a spatial concept that a docstring/comment cites in
place of describing it in prose (AGENTS.md: "make a PNG ... and cite that figure as a comment").
Schematic and not to scale; constants match the shipped ``SimConfig`` defaults (terminal_radius
90 m, corridor 60 x 30 m, ceiling 125 m, hex ``x = R√3(q + r/2)``, ``y = 1.5R·r``). ``context/`` is
unmaintained (AGENTS.md), so this is a single throwaway generator, not per-figure modules.

Run: ``uv run python context/figures/make_figures.py``   (writes every PNG beside this file)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, RegularPolygon  # noqa: E402

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


FIGURES = (
    fig_enroute_rulers, fig_corridor_box_extension, fig_exit_radius, fig_segment_overlaps_column,
    fig_fold_corners, fig_altitude_ladder, fig_segment_frame, fig_hub_placement,
    fig_hex_lattice_overhead, fig_read_envelope,
    fig_search_window, fig_hex_layout, fig_rasterisation_coverage, fig_cell_blocking,
    fig_takeoff_fan, fig_batched_turns, fig_milp_obstacles, fig_hover_tail_steps,
)


def main() -> None:
    """Render every figure in :data:`FIGURES` to ``context/figures/``."""
    for fig_fn in FIGURES:
        fig_fn()


if __name__ == "__main__":
    main()
