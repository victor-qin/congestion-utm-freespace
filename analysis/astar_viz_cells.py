"""Visualize what the A* planner treats as *blocked* in the wall scenario.

A* never sees the wall geometry: it rasterizes every committed Volume4D into a set of
``(q, r, step)`` hex cells (``hexgrid.rasterize_volume``), conservatively inflating each obstacle by
``corridor_width/2 + R`` (one hex circumradius). This script rebuilds that blocked-set *exactly the
way the planner does*, plans the route through it, and draws all three layers on the hex lattice:

  • the true wall box footprint (what you placed),
  • the hexes A* marks blocked (what it actually avoids — fatter and quantized),
  • the A* route centerline (how it detours around the blocked hexes).

This is the "wall scenario" from ``tests/test_astar.py`` made adjustable, so you can slide the wall
around and watch how the conservative rasterization reinterprets it — no debugger needed. Run e.g.:

    uv run python analysis/astar_viz_cells.py
    uv run python analysis/astar_viz_cells.py --x0 800 --y0 -300 --x1 1200 --y1 300
    uv run python analysis/astar_viz_cells.py --thickness 120 --out analysis/thick_wall.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window (matches the repo's viz)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, RegularPolygon  # noqa: E402

from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.geometry import BoxSpec, box_from_segment  # noqa: E402
from freespace_sim.ledger import ReservationLedger  # noqa: E402
from freespace_sim.planner import hexgrid as hg  # noqa: E402
from freespace_sim.planner.astar import AStarPlanner  # noqa: E402
from freespace_sim.types import FlightRequest, IntentStatus, vec  # noqa: E402
from freespace_sim.volumes import Volume4D  # noqa: E402


def build_wall(args, cfg: SimConfig) -> Volume4D:
    """The adjustable analogue of ``tests/test_astar.py::_wall`` — a box bounding a segment.

    Endpoints (x0,y0)→(x1,y1) at cruise altitude, lateral ``thickness`` and vertical ``height``.
    Active for the whole horizon (a permanent wall), exactly like the test's ``_wall``.
    """
    z = cfg.cruise_level_m
    spec = box_from_segment(
        vec(args.x0, args.y0, z), vec(args.x1, args.y1, z), args.thickness, args.height
    )
    return Volume4D(spec, 0.0, 1e6)


def blocked_cells(ledger: ReservationLedger, cfg: SimConfig, R: float, step: int | None):
    """Rebuild A*'s blocked-set the way ``AStarPlanner.plan`` does (astar.py:64-67).

    Returns ``{(q, r): {steps...}}``. With ``step=None`` we keep every cell blocked at any step
    (the right view for a permanent wall); pass a concrete step to inspect a time-windowed obstacle.
    """
    cells: dict[tuple[int, int], set[int]] = {}
    for _fid, vol in ledger.iter_committed():
        for q, r, s in hg.rasterize_volume(vol, cfg, R):
            if step is not None and s != step:
                continue
            cells.setdefault((q, r), set()).add(s)
    return cells


def footprint_xy(spec: BoxSpec) -> np.ndarray:
    """The four xy corners of an oriented box's horizontal footprint (ignores its z extent)."""
    c = np.array(spec.center, float)[:2]
    rot = spec.rotation()
    ax = rot[:2, 0] * spec.extents[0] / 2.0  # local +x half-edge, projected to xy
    ay = rot[:2, 1] * spec.extents[1] / 2.0  # local +y half-edge, projected to xy
    return np.array([c - ax - ay, c + ax - ay, c + ax + ay, c - ax + ay])


def view_window(origin, dest, wall_xy, centerline, margin=300.0):
    """A bounding box around everything worth seeing, padded by ``margin`` metres."""
    pts = [origin[:2], dest[:2], *wall_xy]
    if centerline:
        pts += [p[:2] for p, _t in centerline]
    pts = np.asarray(pts, float)
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    return lo, hi


def hexes_in_window(lo, hi, R):
    """Yield every axial (q, r) whose centre lies inside the view window."""
    qs, rs = [], []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q)
            rs.append(r)
    for q in range(min(qs) - 2, max(qs) + 3):
        for r in range(min(rs) - 2, max(rs) + 3):
            cx, cy = hg.hex_center(q, r, R)
            if lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]:
                yield q, r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x0", type=float, default=1000.0, help="wall segment start x")
    p.add_argument("--y0", type=float, default=-200.0, help="wall segment start y")
    p.add_argument("--x1", type=float, default=1000.0, help="wall segment end x")
    p.add_argument("--y1", type=float, default=200.0, help="wall segment end y")
    p.add_argument("--thickness", type=float, default=40.0, help="wall lateral thickness (m)")
    p.add_argument("--height", type=float, default=400.0, help="wall vertical extent (m)")
    p.add_argument("--origin", type=float, nargs=2, default=[0.0, 0.0], help="flight origin x y")
    p.add_argument("--dest", type=float, nargs=2, default=[2000.0, 0.0], help="flight dest x y")
    p.add_argument(
        "--step", type=int, default=None,
        help="show only cells blocked AT this step (default: union over all steps)",
    )
    p.add_argument("--out", default="analysis/astar_blocked.png", help="output PNG path")
    args = p.parse_args()

    cfg = SimConfig()
    R = hg.circumradius(cfg)
    origin = vec(args.origin[0], args.origin[1], 0.0)
    dest = vec(args.dest[0], args.dest[1], 0.0)

    # 1. commit the wall, exactly like the test does
    ledger = ReservationLedger(cfg)
    wall = build_wall(args, cfg)
    ledger.commit(99, [wall])

    # 2. plan through it
    req = FlightRequest(1, origin, dest, 0.0)
    intent = AStarPlanner().plan(req, ledger, cfg)

    # 3. reconstruct what A* considers blocked + grab the route
    cells = blocked_cells(ledger, cfg, R, args.step)
    centerline = list(intent.centerline) if intent.status is IntentStatus.ACCEPTED else []
    wall_xy = footprint_xy(wall.shape)

    # 4. draw
    lo, hi = view_window(origin, dest, wall_xy, centerline)
    fig, ax = plt.subplots(figsize=(11, 9))
    for q, r in hexes_in_window(lo, hi, R):
        cx, cy = hg.hex_center(q, r, R)
        is_blocked = (q, r) in cells
        ax.add_patch(
            RegularPolygon(
                (cx, cy), numVertices=6, radius=R, orientation=0.0,
                facecolor="firebrick" if is_blocked else "none",
                edgecolor="firebrick" if is_blocked else "0.85",
                alpha=0.55 if is_blocked else 1.0, linewidth=0.5,
            )
        )

    ax.add_patch(Polygon(wall_xy, closed=True, facecolor="none",
                         edgecolor="black", linewidth=2.0, label="true wall box"))

    if centerline:
        xy = np.array([p[:2] for p, _t in centerline])
        ax.plot(xy[:, 0], xy[:, 1], "-o", color="tab:blue", ms=3, lw=1.8,
                label=f"A* route (detour {intent.air_detour_m:.0f} m)")

    ax.plot(*origin[:2], "g^", ms=12, label="origin")
    ax.plot(*dest[:2], "g*", ms=16, label="dest")

    status = intent.status.name
    step_note = "all steps" if args.step is None else f"step {args.step}"
    ax.set_title(
        f"A* blocked cells — wall ({args.x0:.0f},{args.y0:.0f})->({args.x1:.0f},{args.y1:.0f}), "
        f"thickness {args.thickness:.0f} m\n"
        f"{len(cells)} hexes blocked ({step_note}) · inflation = "
        f"{cfg.corridor_width_m / 2 + R:.0f} m · plan: {status}"
    )
    ax.set_xlabel("ENU x (m)")
    ax.set_ylabel("ENU y (m)")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"plan status  : {status}")
    if intent.status is IntentStatus.ACCEPTED:
        print(f"air detour   : {intent.air_detour_m:.1f} m   ground delay: {intent.ground_delay_s:.1f} s")
    print(f"blocked hexes: {len(cells)}   (inflation {cfg.corridor_width_m / 2 + R:.1f} m)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
