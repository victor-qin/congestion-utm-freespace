"""What ``aft +0 / fore +0`` actually looks like: the box IS the hop, centre to centre.

No longitudinal extension at all, so consecutive boxes meet exactly at the waypoints (hex centres)
with zero overlap.  It is the natural "surely this is the minimal box" guess, and it does not help:
each box still straddles a cell boundary -- covering the outer half of the cell it leaves and the
inner half of the cell it enters -- so both incident hops still claim the centre cell and the window
stays at ``(-2, 1)``.

    uv run python analysis/draw_zero_extension.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT not in Path(freespace_sim.__file__).resolve().parents:
    raise SystemExit("loaded the wrong tree")

import probe_cell_box_designs as cellbox  # noqa: E402  (siblings, same directory)
import probe_turn_overlap as turnprobe  # noqa: E402
from freespace_sim import viz  # noqa: E402
from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.planner.hexgrid import circumradius, hex_center  # noqa: E402

SURFACE, INK, INK2, GRID = "#faf9f5", "#141413", "#6b6a66", "#d8d6d0"
BLUE, TEAL, RED, PURPLE = "#4a7fb5", "#2f6d5c", "#c2381f", "#7a5ea8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "analysis" / "zero_extension.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, RegularPolygon

    cfg = SimConfig()
    radius = circumradius(cfg)
    pitch, dt, buf = cfg.corridor_segment_len_m, cfg.dt_s, cfg.time_buffer_s
    r = pitch / 2.0
    path = turnprobe.turn_path(cfg, radius)
    boxes = turnprobe.build(path, cfg, 0.0, 0.0, pads=(buf, buf))

    gap60, pts = turnprobe.uncovered(path, boxes, cfg)
    gap120, _ = turnprobe.uncovered(
        turnprobe.turn_path(cfg, radius, 120.0),
        turnprobe.build(turnprobe.turn_path(cfg, radius, 120.0), cfg, 0.0, 0.0,
                        pads=(buf, buf)), cfg)
    # pinned to the legacy symmetric pad so the comparison is purely geometric
    straight = cellbox.hop_boxes(aft=0.0, fore=0.0, trail=buf, lead=buf, pitch=pitch, dt=dt)
    lo, hi, w = cellbox.window(cellbox.claimed_periods(straight, -r, r, dt))
    early, late = cellbox.margins_s(straight, -r, r, dt, pitch)
    print(f"aft +0 / fore +0:  window ({lo},{hi})  W={w}  headway "
          f"{cellbox.headway_s(straight, -r, r):.1f}s  early {early:.0f}s / late {late:.0f}s  "
          f"turn gap {gap60:.1f} m (60°) / {gap120:.1f} m (120°)")

    fig, ax = plt.subplots(figsize=(10.4, 8.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for q in range(-4, 3):
        for rr in range(-2, 4):
            xy = hex_center(q, rr, radius)
            ax.add_patch(RegularPolygon((float(xy[0]), float(xy[1])), 6, radius=radius,
                                        orientation=0.0, facecolor="#efedE7", edgecolor=GRID,
                                        linewidth=1.0, linestyle=(0, (4, 3)), zorder=0))
    for p in path[1:-1]:
        ax.add_patch(RegularPolygon((float(p[0]), float(p[1])), 6, radius=radius, orientation=0.0,
                                    facecolor=PURPLE, alpha=0.07, edgecolor=PURPLE,
                                    linewidth=1.3, zorder=1))
    for i, (spec, _t0, _t1) in enumerate(boxes):
        colour = (BLUE, TEAL)[i % 2]
        ax.add_patch(Polygon(viz.box_footprint(spec), closed=True, facecolor=colour, alpha=0.32,
                             edgecolor=colour, linewidth=1.8, zorder=3))
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    ax.plot(xs, ys, "-", color=INK, linewidth=2.4, zorder=6)
    ax.plot(xs, ys, "o", color=INK, markersize=7, zorder=7)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=RED, linewidth=7.0,
                solid_capstyle="butt", zorder=8)

    joint = path[2]
    ax.annotate("boxes meet at the WAYPOINT,\nnot at the cell boundary —\nzero overlap, and each box\n"
                "straddles two cells",
                (float(joint[0]), float(joint[1])), textcoords="offset points", xytext=(-40, 118),
                fontsize=10.5, color=INK, ha="center", zorder=9,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.6))
    ax.annotate("cell boundary sits\nmid-box", (float(joint[0]) - r, float(joint[1])),
                textcoords="offset points", xytext=(-30, -110), fontsize=10.5, color=PURPLE,
                ha="center", zorder=9,
                arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.6))

    ax.set_title(
        f"aft +0 / fore +0 — the box IS the hop, hex centre to hex centre\n"
        f"straight-lane window ({lo}, {hi}),  W = {w},  headway "
        f"{cellbox.headway_s(straight, -r, r):.0f} s  —  unchanged from today\n"
        f"60° turn gap {gap60:.0f} m,  120° turn gap {gap120:.0f} m",
        fontsize=12, color=INK, loc="left", pad=11, linespacing=1.5)
    cx, cy = float(path[2][0]), float(path[2][1])
    ax.set_xlim(cx - 250, cx + 250)
    ax.set_ylim(cy - 250, cy + 250)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.text(0.02, 0.030,
             "Alternating box colours; purple hexes are the cells the route passes through.  Each "
             "box runs hex centre to hex centre, so it covers the outer half of",
             fontsize=9.2, color=INK2)
    fig.text(0.02, 0.008,
             "the cell it leaves and the inner half of the one it enters — which is why both "
             "incident hops still claim the middle cell and W stays 4.",
             fontsize=9.2, color=INK2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.095)
    fig.savefig(args.figure, dpi=190, facecolor=SURFACE)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
