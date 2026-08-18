"""How asymmetric filing looks ON THE LEDGER at a 60-degree turn.

Asymmetric (leading-only) filing is what production ships as of 2026-08-14; "sym" below is the
symmetric-pad filing it replaced, retained here because panel 3 is a comparison of the two.

Three panels, all from the repo's real geometry:

1. Plan view of a 60-degree turn at cell C: the inbound and outbound hop boxes, their
   overlap wedge, and a raster proof (every 0.5 m sample of both boxes mapped through
   ``enu_to_axial``) of exactly which hex cells the boxes touch.
2. The ledger timeline at the turn cell under asymmetric filing: the two incident box
   windows, their union (= the 3 claimed rows), the wedge where both are live, and the
   earliest legal crossing traffic -- windows touch exactly, half-open, on the integer
   retime clock.
3. The same-lane minimum headway both filings admit: the binding conflict pair is the
   wedge (follower's inbound box vs leader's outbound box at the shared cell), and the
   capacity rows are exactly tight against it in BOTH filings.

    uv run python analysis/draw_ledger_turn.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.hexgrid import circumradius, enu_to_axial, hex_center  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"

B, C, D = (0, 0), (1, 0), (1, 1)  # hex-adjacent, and the two hop directions differ by 60 deg


def hex_vertices(center: np.ndarray, radius: float, neighbor_angle: float) -> np.ndarray:
    angles = neighbor_angle + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], 1)


def oriented_box(p0: np.ndarray, p1: np.ndarray, half_w: float, ext: float) -> np.ndarray:
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    a, b = p0 - u * ext, p1 + u * ext
    return np.stack([a + n * half_w, b + n * half_w, b - n * half_w, a - n * half_w])


def box_samples(corners: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Dense interior samples of a plan-view box, for the touched-cells raster proof."""

    origin, x_corner, _far, y_corner = corners[0], corners[1], corners[2], corners[3]
    ex, ey = x_corner - origin, y_corner - origin
    nx = max(2, int(np.linalg.norm(ex) / step))
    ny = max(2, int(np.linalg.norm(ey) / step))
    fx, fy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    return origin + fx[..., None] * ex + fy[..., None] * ey


def timeline(ax, dt: float, buf: float, t_lo: float, t_hi: float, rows: list, title: str,
             v_of_t=None) -> None:
    ax.set_facecolor(SURFACE)
    for j in range(int(t_lo // dt), int(t_hi // dt) + 1):
        ax.axvline(j * dt, color=GRID, linewidth=0.8, zorder=1)
        if v_of_t is not None and j * dt < t_hi:
            ax.annotate(f"v={v_of_t + j}", (j * dt + dt / 2, len(rows) - 0.32), fontsize=7.5,
                        color=INK2, ha="center")
    for k, (label, bars) in enumerate(rows):
        y = len(rows) - 1 - k
        for (t0, t1, color, alpha, hatched) in bars:
            ax.fill_betweenx([y - 0.30, y + 0.30], t0, t1, facecolor="none" if hatched else color,
                             edgecolor=color if hatched else "none", hatch="///" if hatched else None,
                             alpha=alpha, linewidth=1.2 if hatched else 0, zorder=3)
            if not hatched:
                ax.plot([t0, t0], [y - 0.30, y + 0.30], color=color, linewidth=1.3, zorder=4)
                ax.plot([t1, t1], [y - 0.30, y + 0.30], color=color, linewidth=1.3, zorder=4)
        ax.annotate(label, (t_lo - 0.6, y), fontsize=8.5, color=INK, ha="right", va="center")
    ax.set_xlim(t_lo - 14.5, t_hi + 0.8)
    ax.set_ylim(-0.62, len(rows) - 0.05)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=8, colors=INK2)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def main() -> None:
    cfg = get_scenario("colgen_test").config()
    radius = circumradius(cfg)
    dt, buf = cfg.dt_s, cfg.time_buffer_s
    half_w = cfg.corridor_width_m / 2.0

    b = np.array(hex_center(*B, radius))
    c = np.array(hex_center(*C, radius))
    d = np.array(hex_center(*D, radius))
    turn_deg = math.degrees(
        math.acos(np.clip(np.dot((c - b) / np.linalg.norm(c - b),
                                 (d - c) / np.linalg.norm(d - c)), -1, 1))
    )
    box_in = oriented_box(b, c, half_w, half_w)
    box_out = oriented_box(c, d, half_w, half_w)

    # Raster proof: which cells do the two boxes actually touch?
    touched: set[tuple[int, int]] = set()
    for corners in (box_in, box_out):
        for point in box_samples(corners).reshape(-1, 2):
            touched.add(enu_to_axial(float(point[0]), float(point[1]), radius))
    extra = sorted(touched - {B, C, D})

    fig = plt.figure(figsize=(14.5, 8.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.46, 0.54), left=0.05, right=0.985,
                            top=0.90, bottom=0.06, hspace=0.42, wspace=0.14)
    ax_plan = fig.add_subplot(grid[:, 0])
    ax_turn = fig.add_subplot(grid[0, 1])
    ax_head = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        f"Asymmetric filing on the ledger at a {turn_deg:.0f}-degree turn "
        "(real corridor_segment_volume geometry, integer retime clock)",
        fontsize=13, color=INK, y=0.965,
    )

    # ------------------------------------------------------------------ plan view
    ax_plan.set_facecolor(SURFACE)
    neighbor_angle = math.atan2((c - b)[1], (c - b)[0])
    ring = {B, C, D}
    for q, r in list(ring):
        for dq in (-1, 0, 1):
            for dr in (-1, 0, 1):
                ring.add((q + dq, r + dr))
    for cell in sorted(ring):
        centre = np.array(hex_center(*cell, radius))
        face = "#f1f0ec" if cell in (B, C, D) else SURFACE
        ax_plan.add_patch(Polygon(hex_vertices(centre, radius, neighbor_angle), closed=True,
                                  facecolor=face, edgecolor=GRID, linewidth=1.0, zorder=1))
    for corners in (box_in, box_out):
        ax_plan.add_patch(Polygon(corners, closed=True, facecolor=BLUE, alpha=0.32,
                                  edgecolor=BLUE, linewidth=1.5, zorder=3))
    for name, centre in (("B", b), ("C", c), ("D", d)):
        ax_plan.plot(*centre, "o", color=INK2, markersize=4, zorder=5)
        ax_plan.annotate(name, centre + np.array([-26, 10]), fontsize=11, color=INK, zorder=6)
    ax_plan.add_patch(Circle(c, radius * math.sqrt(3) / 2, fill=False, edgecolor=ORANGE,
                             linestyle=(0, (4, 3)), linewidth=1.4, zorder=4))
    wedge_reach = half_w * math.sqrt(2.0)
    ax_plan.annotate(
        f"overlap wedge: every extension corner is ≤ {wedge_reach:.0f} m from C\n"
        f"(dashed = C's inradius {radius * math.sqrt(3) / 2:.0f} m) → the wedge stays inside cell C",
        c + np.array([-320, -196]), fontsize=9, color=INK,
    )
    ax_plan.annotate(
        "raster proof (0.5 m grid through both boxes → enu_to_axial):\n"
        f"cells touched = B, C, D{' + ' + str(extra) if extra else ' and NOTHING else'}",
        c + np.array([-320, -246]), fontsize=9, color=INK,
    )
    ax_plan.set_title("Plan view at the turn cell: the two hop boxes and their wedge",
                      fontsize=10.5, color=INK, loc="left")
    ax_plan.set_aspect("equal")
    ax_plan.set_xlim(c[0] - 330, c[0] + 330)
    ax_plan.set_ylim(c[1] - 270, c[1] + 290)
    ax_plan.set_xlabel("east (m)", fontsize=8, color=INK2)
    ax_plan.tick_params(labelsize=8, colors=INK2)
    for spine in ax_plan.spines.values():
        spine.set_color(GRID)

    # -------------------------------------------- timeline at the turn cell (asym)
    # Visits: B at v=0, C at v=1, D at v=2.  Asym windows: inbound [0, 8], outbound [4, 12].
    timeline(
        ax_turn, dt, buf, t_lo=-buf, t_hi=5 * dt + 2,
        v_of_t=0,
        rows=[
            ("inbound box B→C", [(-buf, 2 * dt, INK2, 0.5, True), (0, 2 * dt, BLUE, 0.32, False)]),
            ("outbound box C→D", [(0 - buf + dt, dt, INK2, 0.5, True), (dt, 3 * dt, BLUE, 0.32, False)]),
            ("both live (wedge)", [(dt, 2 * dt, ORANGE, 0.45, False)]),
            ("rows cell C claims", [(0, dt, AQUA, 0.30, False), (dt, 2 * dt, AQUA, 0.30, False),
                                    (2 * dt, 3 * dt, AQUA, 0.30, False)]),
            ("earliest crossing flight at C", [(3 * dt, 5 * dt, INK2, 0.30, False)]),
        ],
        title="Ledger timeline at turn cell C (asym): window union [0,12] s = the 3 claimed rows",
    )
    ax_turn.annotate("touch at t=12, half-open\n⇒ no conflict (exact k·dt)",
                     (3 * dt + 0.5, 0.42), fontsize=8, color=INK)
    ax_turn.annotate("hatched = the trailing pad sym adds", (-buf - 14.2, 3.5), fontsize=7.5,
                     color=INK2, va="center")
    ax_turn.set_xlabel("time (s); C visited at v=1 (t=4)", fontsize=8.5, color=INK2)

    # ----------------------------------------------- same-lane minimum headway
    # Leader visits C at v=1; follower on the same lane B→C→D.  Binding ledger pair:
    # follower's inbound B→C box vs leader's outbound C→D box (they overlap in the wedge).
    timeline(
        ax_head, dt, buf, t_lo=-buf, t_hi=7 * dt,
        rows=[
            ("sym: leader C→D box", [(0, 3 * dt, BLUE, 0.32, False)]),
            ("sym: follower B→C box (Δv=4)", [(3 * dt, 6 * dt, ORANGE, 0.32, False)]),
            ("asym: leader C→D box", [(dt, 3 * dt, BLUE, 0.32, False)]),
            ("asym: follower B→C box (Δv=3)", [(3 * dt, 5 * dt, ORANGE, 0.32, False)]),
        ],
        title="Minimum same-lane headway: the rows (width 4 vs 3) equal the ledger's minimum",
    )
    ax_head.annotate("windows touch at t=12 in both filings;\nheadway at C: 16 s (sym) → 12 s (asym)",
                     (3 * dt + 1.0, 3.42), fontsize=8, color=INK, va="top")
    ax_head.set_xlabel("time (s); leader visits C at t=4, wedge pair is the binding conflict",
                       fontsize=8.5, color=INK2)

    out = REPO_ROOT / "analysis" / "colgen_ledger_turn.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"touched cells: {sorted(touched)}  (extra beyond path: {extra or 'none'})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
