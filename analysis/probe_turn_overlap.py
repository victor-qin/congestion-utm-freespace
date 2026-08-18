"""Does an ``aft/fore`` slide survive a 60-degree lattice turn, and what does it cost the window?

A slide that measures beautifully on a straight lane can still leave the aircraft outside its own
reservation the moment the lane bends: the box covering a cell is aligned with the OUTBOUND leg, so
a turning inbound approach swings past its lateral half-width almost immediately, while the
previous box has been pulled back off the shared boundary.

Both constraints are measured here against the shipped geometry:

* turn coverage -- the aircraft's own polyline is walked through a real 60-degree hex turn and
  tested against real ``box_from_segment`` boxes, reporting the longest stretch inside no live box;
* window cost -- the same ``(aft, fore)`` pair run through the straight-lane row model in
  ``probe_cell_box_designs``.

    uv run python analysis/probe_turn_overlap.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT not in Path(freespace_sim.__file__).resolve().parents:
    raise SystemExit("loaded the wrong tree")

import probe_cell_box_designs as cellbox  # noqa: E402  (sibling module, same directory)
from freespace_sim import viz  # noqa: E402
from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.geometry import box_from_segment  # noqa: E402
from freespace_sim.planner.hexgrid import AXIAL_NEIGHBORS, circumradius, hex_center  # noqa: E402

SURFACE, INK, INK2, GRID = "#faf9f5", "#141413", "#6b6a66", "#d8d6d0"
BLUE, ORANGE, RED, GREEN = "#4a7fb5", "#d97757", "#c2381f", "#3f7d5a"


def _pt(cell, cfg: SimConfig, radius: float) -> np.ndarray:
    xy = hex_center(*cell, radius)
    return np.array([float(xy[0]), float(xy[1]), cfg.cruise_level_m])


def turn_path(cfg: SimConfig, radius: float, degrees: float = 60.0) -> list[np.ndarray]:
    """Hex centres turning by ``degrees`` at index 3, with two straight hops on either side.

    The turn must be INTERIOR to the polyline: the last box of any retracted design stops short of
    the final waypoint, so a path that ends at the turn reports that tail as a coverage gap and the
    turn's own behaviour is never measured.
    """
    straight = [(-3, 0), (-2, 0), (-1, 0), (0, 0)]
    inbound = _pt((0, 0), cfg, radius) - _pt((-1, 0), cfg, radius)
    inbound /= np.linalg.norm(inbound)
    for dq, dr in AXIAL_NEIGHBORS:
        if (dq, dr) == (-1, 0):
            continue
        out = _pt((dq, dr), cfg, radius) - _pt((0, 0), cfg, radius)
        out /= np.linalg.norm(out)
        if abs(math.degrees(math.acos(float(np.clip(inbound @ out, -1.0, 1.0)))) - degrees) < 1e-6:
            cells = straight + [(dq, dr), (2 * dq, 2 * dr), (3 * dq, 3 * dr)]
            return [_pt(c, cfg, radius) for c in cells]
    raise RuntimeError(f"no {degrees} degree neighbour found")


def straight_path(cfg: SimConfig, radius: float) -> list[np.ndarray]:
    """The same length of lane with no turn -- the control for the turn measurement."""
    return [_pt((q, 0), cfg, radius) for q in range(-3, 4)]


def build(path: list[np.ndarray], cfg: SimConfig, aft: float, fore: float,
          pads: tuple[float, float] | None = None):
    """The real ledger boxes for this design: ``[p0 - aft*u, p1 + fore*u]`` with the hop's window.

    ``pads`` is ``(trail, lead)`` in seconds; it defaults to whatever filing is checked out, read
    off ``corridor_segment_volume`` rather than assumed, since this tree has shipped both a
    symmetric and a leading-only pad.
    """
    trail, lead = cellbox.measured_pads(cfg) if pads is None else pads
    out = []
    for k in range(len(path) - 1):
        p0, p1 = path[k], path[k + 1]
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        spec = box_from_segment(p0 - u * aft, p1 + u * fore,
                                cfg.corridor_width_m, cfg.corridor_height_m)
        out.append((spec, k * cfg.dt_s - trail, (k + 1) * cfg.dt_s + lead))
    return out


def _inside(spec, p: np.ndarray) -> bool:
    loc = spec.rotation().T @ (p - np.array(spec.center, float))   # rot columns are the local axes
    return all(abs(loc[i]) <= spec.extents[i] / 2.0 + 1e-9 for i in range(3))


def uncovered(path, boxes, cfg: SimConfig, samples: int = 20001):
    """``(longest_gap_m, [(x, y), ...])`` for the aircraft's own polyline against its own boxes.

    Only the INTERIOR is walked -- the first and last hop are skipped, because a retracted design's
    end boxes legitimately stop short of the route's endpoints (that stretch is the terminal
    cylinder's job, not a turn defect) and would otherwise mask the measurement.
    """
    lens = [float(np.linalg.norm(path[k + 1] - path[k])) for k in range(len(path) - 1)]
    total = sum(lens)
    s_lo, s_hi = lens[0], total - lens[-1]
    worst = run = 0.0
    pts: list[tuple[float, float]] = []
    for i in range(samples):
        s = s_lo + (s_hi - s_lo) * i / (samples - 1)
        acc = 0.0
        for k, seg in enumerate(lens):
            if s <= acc + seg or k == len(lens) - 1:
                f = (s - acc) / seg
                p = path[k] + (path[k + 1] - path[k]) * f
                t = (k + f) * cfg.dt_s
                break
            acc += seg
        if any(t0 <= t < t1 and _inside(spec, p) for spec, t0, t1 in boxes):
            run = 0.0
        else:
            run += (s_hi - s_lo) / (samples - 1)
            pts.append((float(p[0]), float(p[1])))
        worst = max(worst, run)
    return worst, pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "analysis" / "turn_overlap.png")
    args = ap.parse_args()

    cfg = SimConfig()
    radius = circumradius(cfg)
    pitch, dt, buf = cfg.corridor_segment_len_m, cfg.dt_s, cfg.time_buffer_s
    r = pitch / 2.0
    path = turn_path(cfg, radius)

    designs = [
        ("TODAY            aft +30 / fore +30", 30.0, 30.0),
        ("slid, no overlap  aft +60 / fore -60", r, -r),
        ("proposed          aft +75 / fore -45", 75.0, -45.0),
        ("                  aft +90 / fore -30", 90.0, -30.0),
        ("                  aft +105 / fore -15", 105.0, -15.0),
    ]

    control = straight_path(cfg, radius)

    print(f"pitch {pitch:.0f} m   inradius {r:.0f} m   corridor width {cfg.corridor_width_m:.0f} m"
          f"   dt {dt:.0f} s\nspatial variants compared at the LEGACY symmetric pad "
          f"(-{buf:.0f}, +{buf:.0f}) s, so only the geometry differs; this tree itself files "
          f"{cellbox.measured_pads(cfg)}\n")
    head = (f"{'design':<38} {'box len':>8} {'straight':>9} {'60° turn':>9} {'window':>9} {'W':>3} "
            f"{'headway':>9}")
    print(head)
    print("-" * len(head))
    results = []
    for name, aft, fore in designs:
        boxes = build(path, cfg, aft, fore, pads=(buf, buf))
        gap, pts = uncovered(path, boxes, cfg)
        flat, _ = uncovered(control, build(control, cfg, aft, fore, pads=(buf, buf)), cfg)
        straight = cellbox.hop_boxes(aft=aft, fore=fore, trail=buf, lead=buf, pitch=pitch, dt=dt)
        lo, hi, w = cellbox.window(cellbox.claimed_periods(straight, -r, r, dt))
        headway = cellbox.headway_s(straight, -r, r)
        print(f"{name:<38} {pitch + aft + fore:>6.0f} m {flat:>7.1f} m {gap:>7.1f} m "
              f"{f'({lo},{hi})':>9} {w:>3} {headway:>7.1f} s")
        results.append((name, aft, fore, boxes, gap, pts, (lo, hi), w))

    draw(args.figure, cfg, radius, path, results)



def draw(out: Path, cfg: SimConfig, radius: float, path, results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, RegularPolygon

    shown = results[:4]
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 7.4), facecolor=SURFACE)
    fig.suptitle(
        "The 60° turn joint, plan view — is there still overlap between the two legs' boxes?\n"
        "real `box_from_segment` output; red is the stretch of the aircraft's own polyline that "
        "lies inside NO live box",
        fontsize=13, color=INK, y=0.968,
    )

    turn = path[3]                                   # the vertex the aircraft turns at
    cells = [(q, r) for q in range(-3, 3) for r in range(-3, 3)]
    shades = ["#4a7fb5", "#2f6d5c", "#4a7fb5", "#2f6d5c", "#4a7fb5", "#2f6d5c"]

    for ax, (name, aft, fore, boxes, gap, pts, win, w) in zip(axes, shown):
        ax.set_facecolor(SURFACE)
        for c in cells:
            xy = hex_center(*c, radius)
            ax.add_patch(RegularPolygon((float(xy[0]), float(xy[1])), 6, radius=radius,
                                        orientation=0.0, facecolor="#efedE7", edgecolor=GRID,
                                        linewidth=1.0, linestyle=(0, (4, 3)), zorder=0))
        # the turn cell, whose capacity rows the window counts
        ax.add_patch(RegularPolygon((float(turn[0]), float(turn[1])), 6, radius=radius,
                                    orientation=0.0, facecolor="#7a5ea8", alpha=0.10,
                                    edgecolor="#7a5ea8", linewidth=1.8, zorder=1))
        for i, (spec, _t0, _t1) in enumerate(boxes):
            ax.add_patch(Polygon(viz.box_footprint(spec), closed=True, facecolor=shades[i % 6],
                                 alpha=0.30, edgecolor=shades[i % 6], linewidth=1.7,
                                 zorder=3 + i * 0.1))
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        ax.plot(xs, ys, "-", color=INK, linewidth=2.4, zorder=6)
        ax.plot(xs, ys, "o", color=INK, markersize=6, zorder=7)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=RED, linewidth=7.0,
                    solid_capstyle="butt", zorder=8)
            mid = pts[len(pts) // 2]
            ax.annotate(f"{gap:.0f} m of the aircraft's\nown path outside\nevery live box", mid,
                        textcoords="offset points", xytext=(-6, 105), fontsize=10, color=RED,
                        weight="bold", ha="center", zorder=9,
                        arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.8))
        else:
            ax.annotate("legs still overlap —\nno gap", (float(turn[0]) - 96, float(turn[1]) + 96),
                        fontsize=10, color=GREEN, weight="bold", ha="center", zorder=9)
        ax.set_title(
            f"{name.strip()}\nbox {cfg.corridor_segment_len_m + aft + fore:.0f} m long,  "
            f"straight-lane window {win},  W = {w}\n"
            f"{'turn: COVERED' if gap <= 1e-6 else f'turn: GAP {gap:.0f} m'}",
            fontsize=10.6, color=INK, loc="left", pad=9, linespacing=1.5,
        )
        ax.set_xlim(float(turn[0]) - 175, float(turn[0]) + 175)
        ax.set_ylim(float(turn[1]) - 175, float(turn[1]) + 175)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)

    fig.text(0.007, 0.078,
             "Alternating box colours: where two boxes overlap the shading doubles.  Purple hex = "
             "the cell whose capacity rows the window counts.",
             fontsize=9.2, color=INK)
    fig.text(0.007, 0.052,
             "A cell's box is aligned with the leg the aircraft LEAVES on, so at a turn the inbound "
             "approach crosses its 30 m half-width immediately — only longitudinal reach from the "
             "PREVIOUS box can cover that stretch.",
             fontsize=9.2, color=INK2)
    fig.text(0.007, 0.026,
             "But that reach is what re-widens the window: fore ≥ −34.6 m is needed to close the "
             "turn gap, while W = 3 needs the box inside one cell, i.e. fore ≤ −60 m.  The two "
             "intervals do not meet.",
             fontsize=9.2, color=INK2)
    fig.text(0.007, 0.002,
             "At a 4 s pad the 12 s claim budget is fully spent by 4 s of dwell + 8 s of pad, "
             "leaving nothing for overlap — so on one box per cell the two requirements are "
             "mutually exclusive.",
             fontsize=9.2, color=INK2)
    fig.subplots_adjust(left=0.032, right=0.99, top=0.815, bottom=0.135, wspace=0.16)
    fig.savefig(out, dpi=185, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
