"""What the replay draws, before and after the move to leading-only reservation pads.

A* plans are bit-identical under both filings (`analysis/probe_physics_500.py` pins the plan
fingerprint), so nothing about the trajectory moves.  What moves is which reservation boxes are
LIVE at a given instant, and therefore what the replay lights up.

Everything here is measured from the repo, not sketched: box footprints come from
`volumes.corridor_segment_volume` via `viz.box_footprint` (the same builder the shipped `segPoly`
JS is pinned against by `test_shipped_segpoly_js_reproduces_the_corridor_builder`), and the
live-window rule is the one `test_shipped_replay_draws_a_segment_from_exactly_its_start_time`
executes in node against the shipped source.

    uv run python analysis/draw_replay_ab.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import freespace_sim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT not in Path(freespace_sim.__file__).resolve().parents:
    raise SystemExit("loaded the wrong tree")

from freespace_sim import viz, volumes  # noqa: E402
from freespace_sim.scenario import scenario_from_requests  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402
from freespace_sim.sim import run as sim_run  # noqa: E402

SURFACE, INK, INK2, GRID = "#faf9f5", "#141413", "#6b6a66", "#d8d6d0"
BLUE, ORANGE, DIM = "#4a7fb5", "#d97757", "#c9c7c1"


def live_range(times, tq, buf, trailing_pad):
    """Inclusive [a, z] of segments live at ``tq``; ``trailing_pad`` is 0 today, ``buf`` before.

    Segment i spans ``[t[i] - trailing_pad, t[i+1] + buf)`` -- live iff
    ``t[i] - trailing_pad <= tq``  and  ``t[i+1] + buf > tq``.
    """
    a, z = None, None
    for i in range(len(times) - 1):
        if times[i] - trailing_pad <= tq and times[i + 1] + buf > tq:
            a = i if a is None else a
            z = i
    return a, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--flights", type=int, default=24)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "analysis" / "colgen_replay_ab.png")
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    result = sim_run(cfg, scenario=scenario_from_requests(requests), demand=demand,
                     planner_name="astar")
    accepted = [i for i in result.intents if i.accepted]
    focus = max(accepted, key=lambda i: len(i.centerline))
    cl = focus.centerline
    times = [float(t) for _p, t in cl]
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p, _t in cl]
    buf = cfg.time_buffer_s

    # An instant mid-cruise, deliberately NOT on a dt boundary so the picture is the generic case.
    tq = times[len(times) // 2] + cfg.dt_s / 2.0
    a_now, z_now = live_range(times, tq, buf, 0.0)
    a_old, z_old = live_range(times, tq, buf, buf)
    print(f"flight {focus.request.flight_id}  t={tq:.1f}s  "
          f"today segments [{a_now},{z_now}] = {z_now - a_now + 1}   "
          f"legacy [{a_old},{z_old}] = {z_old - a_old + 1}")

    # Drone position: linear interpolation, exactly what the replay's posAt does.
    k = max(0, min(len(times) - 2, int(np.searchsorted(times, tq, side="right")) - 1))
    u = (tq - times[k]) / (times[k + 1] - times[k])
    drone = tuple(pts[k][j] + (pts[k + 1][j] - pts[k][j]) * u for j in range(3))

    lo_i, hi_i = min(a_now, a_old) - 3, max(z_now, z_old) + 3
    lo_i, hi_i = max(0, lo_i), min(len(times) - 2, hi_i)

    fig = plt.figure(figsize=(14.6, 6.4), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.22), left=0.035, right=0.985,
                            top=0.845, bottom=0.145, wspace=0.16)
    fig.suptitle(
        "What the replay lights up at one instant — same flight, same second, same plan\n"
        f"A* plans are bit-identical under both filings; only the live-window rule moved "
        f"(flight {focus.request.flight_id}, t = {tq:.0f} s)",
        fontsize=12.5, color=INK, y=0.965,
    )

    extra = set(range(a_old, z_old + 1)) - set(range(a_now, z_now + 1))   # lit only BEFORE

    def plan_panel(ax, a, z, title, note, mark_extra=False):
        ax.set_facecolor(SURFACE)
        for i in range(lo_i, hi_i + 1):
            spec_i = volumes.corridor_segment_volume(
                np.array(pts[i]), times[i], np.array(pts[i + 1]), times[i + 1], cfg
            ).shape
            poly = viz.box_footprint(spec_i)
            lit = a is not None and a <= i <= z
            odd = mark_extra and i in extra
            ax.add_patch(Polygon(poly, closed=True,
                                 facecolor=(ORANGE if odd else BLUE) if lit else "none",
                                 alpha=0.45 if lit else 1.0,
                                 edgecolor=(ORANGE if odd else BLUE) if lit else DIM,
                                 linewidth=2.0 if odd else (1.5 if lit else 0.9),
                                 linestyle="-" if lit else (0, (3, 3)), zorder=4 if odd else (3 if lit else 2)))
            if odd:
                c = np.asarray(poly).mean(axis=0)
                ax.annotate("reserved 4 s\nbefore the drone\nreaches it", (c[0], c[1]),
                            textcoords="offset points", xytext=(-96, -34), fontsize=8.5,
                            color=ORANGE, ha="center", zorder=7,
                            arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2))
        xs = [p[0] for p in pts[lo_i:hi_i + 2]]
        ys = [p[1] for p in pts[lo_i:hi_i + 2]]
        ax.plot(xs, ys, "-", color=INK2, linewidth=1.0, zorder=4)
        ax.plot(*drone[:2], "o", color=INK, markersize=8, zorder=6)
        ax.annotate("drone", (drone[0], drone[1]), textcoords="offset points", xytext=(9, -14),
                    fontsize=9, color=INK, zorder=6)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.annotate(note, (0.5, -0.085), xycoords="axes fraction", ha="center",
                    fontsize=9.5, color=INK)
        cx, cy = drone[0], drone[1]
        ax.set_xlim(cx - 330, cx + 330)
        ax.set_ylim(cy - 330, cy + 330)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)

    ax_now = fig.add_subplot(grid[0, 0])
    ax_old = fig.add_subplot(grid[0, 1])
    plan_panel(ax_old, a_old, z_old,
               f"BEFORE — symmetric pad  [tᵢ−{buf:.0f}, tᵢ₊₁+{buf:.0f})",
               f"{z_old - a_old + 1} boxes lit  (boxes {a_old}–{z_old})", mark_extra=True)
    plan_panel(ax_now, a_now, z_now,
               f"TODAY — leading-only pad  [tᵢ, tᵢ₊₁+{buf:.0f})",
               f"{z_now - a_now + 1} boxes lit  (boxes {a_now}–{z_now}) — the drone is inside box {z_now}")

    # ------------------------------------------------------------------ space-time
    ax = fig.add_subplot(grid[0, 2])
    ax.set_facecolor(SURFACE)
    span = range(max(0, a_now - 2), min(len(times) - 2, z_old + 2) + 1)
    for i in span:
        for row, (trailing, colour, label) in enumerate(
            ((buf, ORANGE, "before"), (0.0, BLUE, "today"))
        ):
            y = i + (0.20 if row else -0.20)
            live = times[i] - trailing <= tq < times[i + 1] + buf
            ax.plot([times[i] - trailing, times[i + 1] + buf], [y, y],
                    color=colour, linewidth=6.5 if live else 3.0,
                    alpha=0.95 if live else 0.30,
                    solid_capstyle="butt", zorder=3)
    ax.axvline(tq, color=INK, linewidth=1.6, zorder=5)
    ax.annotate(f"now  t={tq:.0f}s", (tq, min(span) - 0.72), fontsize=9, color=INK,
                ha="center", va="top")
    for i in span:
        ax.axhline(i, color=GRID, linewidth=0.6, zorder=1)
    ax.set_yticks(list(span))
    ax.set_yticklabels([f"box {i}" for i in span], fontsize=8)
    ax.set_ylim(min(span) - 1.15, max(span) + 0.7)
    ax.set_xlabel("time (s) — each bar is one box's live window", fontsize=9, color=INK2)
    ax.set_title("Why: the trailing pad opened the NEXT box early", fontsize=11, color=INK,
                 loc="left")
    ax.invert_yaxis()
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ax.spines.values():
        s.set_color(GRID)
    handles = [
        plt.Line2D([], [], color=ORANGE, linewidth=6.5, label="before: [tᵢ−buf, tᵢ₊₁+buf)"),
        plt.Line2D([], [], color=BLUE, linewidth=6.5, label="today: [tᵢ, tᵢ₊₁+buf)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.075), ncol=2,
              fontsize=8.5, frameon=False)

    fig.text(0.035, 0.030,
             f"Boxes are the reservation the ledger actually holds, drawn by the shipped builder "
             f"(corridor {cfg.corridor_width_m:.0f} m wide, {cfg.corridor_segment_len_m:.0f} m pitch, "
             f"dt = {cfg.dt_s:.0f} s, time_buffer_s = {buf:.0f} s).",
             fontsize=8.8, color=INK2)
    fig.text(0.035, 0.006,
             "The removed trailing pad was one dt, so one fewer box lights AHEAD of the drone: "
             "airspace is no longer held before the aircraft reaches it. Nothing behind it changed.",
             fontsize=8.8, color=INK2)

    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
