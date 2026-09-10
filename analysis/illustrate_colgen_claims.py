"""Draw ledger geometry and occupancy from the actual A*/CG builders.

This is a controlled illustration of the SAME timed route, not an A* vs CG
optimization benchmark. The figures isolate two interior hops so endpoint
cylinders and other route segments do not hide the cell-window construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar.occupancy import HexOccupancyService
from freespace_sim.planner.astar.planner import AStarPlanner
from freespace_sim.planner.colgen.network import build_flight_graph, column_claims
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.translate import Column, column_to_intent
from freespace_sim.planner.colgen.windows import derive_cell_window, visit_rows
from freespace_sim.types import FlightRequest, IntentStatus


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/colgen_claims_20260909"
INPUT = ROOT / (
    "analysis/colgen_cheap_pricing_1200s_20260909/"
    "density_faa_wing_zipline_seed0_iteration_ip_eager/inputs.json"
)
BLUE = "#276b9c"
TEAL = "#258d92"
ORANGE = "#c96a20"
DARK = "#172e40"
GREY = "#637582"
GREEN = "#397c62"


def construct():
    raw = json.loads(INPUT.read_text())["config"]
    cfg = SimConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    cells = ((-3, 0), (-2, 0), (-1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1))
    radius = hg.circumradius(cfg)
    points = [np.r_[hg.hex_center(*q, radius), cfg.flight_levels_m[0]] for q in cells]
    request = FlightRequest(
        0, np.r_[points[0][:2], 0.0], np.r_[points[-1][:2], 0.0], 0.0, 8.0,
    )
    column = Column(
        flight_id=0, departure_step=2, level=0, origin_lane_idx=None,
        dest_lane_idx=None, cell_path=cells, delay_s=0.0,
    )
    graph = build_flight_graph(request, cfg, (), ColGenParams())
    intent = column_to_intent(column, request, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    full_claims = column_claims(column, graph, cfg)
    start = (column.departure_step + cfg.climb_steps_to(cfg.flight_levels_m[0])) * cfg.dt_s
    waypoints = [(point, start + i * cfg.dt_s) for i, point in enumerate(points)]
    astar_volumes, _, _, _ = AStarPlanner(compiled=False)._build(
        waypoints, request.origin, request.dest, column.departure_step, 0, cfg,
    )
    assert len(astar_volumes) == len(intent.volumes)
    # For this selected route, the actual builders return exactly matching
    # shapes, times and tags, including the unplotted endpoint cylinders.
    for av, cv in zip(astar_volumes, intent.volumes, strict=True):
        assert av == cv
    excerpt = astar_volumes[3:5]
    assert all(isinstance(v.shape, BoxSpec) for v in excerpt)
    assert [(v.t_start, v.t_end) for v in excerpt] == [(32.0, 44.0), (36.0, 48.0)]
    assert all(v.shape.extents == (180.0, 60.0, 30.0) for v in excerpt)
    excerpt_cells = cells[2:5]
    visits = (9, 10, 11)
    offsets = derive_cell_window(cfg)
    assert offsets == (-2, 1)
    excerpt_claims = {
        (q, 0, tau)
        for q, visit in zip(excerpt_cells, visits, strict=True)
        for tau in visit_rows(visit, offsets)
    }
    b_rows = {tau for q, level, tau in excerpt_claims if q == (0, 0)}
    assert b_rows == {
        r.step for r in full_claims if r.kind == "cell" and r.cell_coord == (0, 0)
    } == {8, 9, 10, 11}
    service = HexOccupancyService(cfg)
    for vol in excerpt:
        service.add_volume(vol)
    steps = range(6, 17)
    astar_blocked = [s for s in steps if service.is_blocked(0, 0, 0, s)]
    cg_blocked = [s for s in steps if b_rows.intersection(visit_rows(s, offsets))]
    assert astar_blocked == list(range(7, 15))
    assert cg_blocked == list(range(7, 14))
    blocked_cells = {(q, r) for q, r, level in service.blocked[10] if level == 0}
    claimed_cells = {q for q, level, tau in excerpt_claims if tau == 10}
    assert len(blocked_cells) == 13
    assert claimed_cells == set(excerpt_cells)
    data = dict(
        config_source=str(INPUT), same_route_builders_identical=True,
        scope="Two interior hops only; surrounding route and endpoint contributions omitted.",
        dt_s=cfg.dt_s, time_buffer_s=cfg.time_buffer_s,
        hex_pitch_m=cfg.corridor_segment_len_m, hex_circumradius_m=radius,
        box_extents_m=list(excerpt[0].shape.extents),
        excerpt_cells=excerpt_cells, excerpt_visit_steps=visits, cell_offsets=offsets,
        ledger_windows_s=[(v.t_start, v.t_end) for v in excerpt],
        cg_b_resource_rows=sorted(b_rows), astar_b_blocked_entry_steps=astar_blocked,
        cg_b_conflicting_visit_steps=cg_blocked,
        astar_blocked_cells_at_step10=sorted(blocked_cells),
        cg_claimed_cells_at_row10=sorted(claimed_cells),
        comparison_note=(
            "A* flags are candidate-entry exclusions; CG rows are source resource occupancy. "
            "The lower two temporal rows compare candidate entry times on both sides, at B only."
        ),
    )
    return cfg, excerpt, excerpt_cells, blocked_cells, claimed_cells, data


def vertices(cell, radius):
    centre = hg.hex_center(*cell, radius)
    angle = np.deg2rad(30 + np.arange(6) * 60)
    return centre + radius * np.column_stack((np.cos(angle), np.sin(angle)))


def box_footprint(spec):
    corners = np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    ], dtype=float) * (np.asarray(spec.extents) / 2)
    return (corners @ spec.rotation().T + np.asarray(spec.center))[:, :2]


def spatial(cfg, volumes, cells, blocked, claimed):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.3))
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.23, top=0.78, wspace=0.16)
    fig.suptitle("From ledger boxes to cell occupancy", fontsize=23, fontweight="bold", y=0.975)
    fig.text(
        0.5, 0.912,
        "Same route: A at 36 s → B at 40 s → C at 44 s  |  120 m hops  |  100 m cruise level",
        ha="center", fontsize=14, color=GREY,
    )
    titles = (
        ("Ledger: A* and CG", "Both builders produce these same boxes"),
        ("A* search occupancy", "13 blocked entry cells at step 10 (40 s)"),
        ("CG master coefficients", "3 claimed cells in row period 10: [40, 44) s"),
    )
    radius = hg.circumradius(cfg)
    for index, ax in enumerate(axes):
        for q in range(-4, 4):
            for r in range(-3, 4):
                cell = (q, r)
                fill, edge, lw = "#ffffff", "#dbe3e9", 0.8
                if index == 1 and cell in blocked:
                    fill, edge, lw = "#d8e9f5", BLUE, 1.0
                if index == 2 and cell in claimed:
                    fill, edge, lw = "#f8ddc3", ORANGE, 1.6
                ax.add_patch(Polygon(vertices(cell, radius), facecolor=fill, edgecolor=edge, lw=lw))
        for vol, colour in zip(volumes, (BLUE, TEAL), strict=True):
            ax.add_patch(Polygon(
                box_footprint(vol.shape), facecolor=colour if index == 0 else "none",
                alpha=0.30 if index == 0 else 0.80, edgecolor=colour, lw=2.2,
            ))
        xy = np.array([hg.hex_center(*q, radius) for q in cells])
        for first, second in zip(xy, xy[1:]):
            ax.annotate("", xy=second, xytext=first,
                        arrowprops=dict(arrowstyle="-|>", lw=2.1, color=DARK, mutation_scale=14))
        ax.scatter(xy[:, 0], xy[:, 1], color=DARK, s=38, zorder=5)
        offsets = ((-15, -36), (16, -36), (19, 13))
        for label, cell, pt, offset in zip("ABC", cells, xy, offsets, strict=True):
            ax.annotate(
                f"{label}  {cell}", xy=pt, xytext=offset, textcoords="offset points",
                fontsize=12, weight="bold", color=DARK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=2),
            )
        ax.set_title(titles[index][0], fontsize=17, fontweight="bold", pad=37)
        ax.text(0.5, 1.03, titles[index][1], transform=ax.transAxes,
                ha="center", fontsize=11.5, color=GREY)
        ax.set(xlim=(-320, 260), ylim=(-190, 300), aspect="equal", xlabel="East (m)")
        ax.set_xticks([-240, -120, 0, 120])
        ax.set_yticks([-120, 0, 120, 240])
        for spine in ax.spines.values():
            spine.set_color("#c1ccd3")
        ax.tick_params(labelsize=10, colors=GREY)
    axes[0].set_ylabel("North (m)", color=GREY)
    axes[0].annotate("", xy=(30, -88), xytext=(-150, -88),
                     arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.1))
    axes[0].text(-60, -110, "180 m box length", ha="center", fontsize=11, color=GREY)
    fig.legend(
        handles=[
            Patch(facecolor="#8db4d0", edgecolor=BLUE, label="Incoming ledger box"),
            Patch(facecolor="#a8d5d5", edgecolor=TEAL, label="Outgoing ledger box"),
            Patch(facecolor="#d8e9f5", edgecolor=BLUE, label="A* blocked-entry footprint"),
            Patch(facecolor="#f8ddc3", edgecolor=ORANGE, label="CG resource claims"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=4, frameon=False, fontsize=12,
    )
    fig.text(
        0.5, 0.085,
        "Boxes: 180 × 60 × 30 m, including a 30 m extension at each end of a 120 m hop.",
        ha="center", fontsize=12, color=DARK,
    )
    fig.text(
        0.5, 0.047,
        "A* entry flags and CG resource rows have different meanings; the temporal figure compares their effects.",
        ha="center", fontsize=11, color=GREY,
    )
    fig.text(
        0.5, 0.015,
        "Controlled two-hop excerpt from a longer valid route. Endpoint cylinders and the rest of the route are omitted.",
        ha="center", fontsize=10.5, color=GREY,
    )
    fig.savefig(OUT / "spatial.png", dpi=180, facecolor="white")
    plt.close(fig)


def temporal(cfg, volumes, data):
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.79, bottom=0.22)
    fig.suptitle("Cell B: ledger windows, resource rows, and blocked entry times",
                 fontsize=21, fontweight="bold", y=0.975)
    fig.text(0.5, 0.926,
             "The reserved route passes B at 40 s (step 10).  Δt = 4 s; each ledger box has ±4 s time buffering.",
             ha="center", fontsize=13, color=GREY)
    ax.set_xlim(23, 65)
    ax.set_ylim(0.1, 7.8)
    ticks = np.arange(24, 65, 4)
    ax.set_xticks(ticks)
    ax.set_xlabel("Time (seconds); lower two rows show candidate visit time at B", fontsize=12, labelpad=10)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 4, lambda s: s * 4))
    sec.set_xticks(ticks / 4)
    sec.set_xlabel("Integer step / row index", fontsize=12, labelpad=9)
    ax.grid(axis="x", color="#dbe3e9", lw=0.9, zorder=0)
    ax.axvline(40, color=DARK, ls=":", lw=1.5)
    y_ledger = (6.8, 5.65)
    for y, volume, colour, name, nominal in zip(
        y_ledger, volumes, (BLUE, TEAL), ("A → B", "B → C"), ((36, 40), (40, 44)), strict=True,
    ):
        ax.add_patch(Rectangle(
            (volume.t_start, y - 0.28), volume.t_end - volume.t_start, 0.56,
            facecolor=colour, alpha=0.25, edgecolor=colour, lw=1.4, zorder=2,
        ))
        ax.add_patch(Rectangle((nominal[0], y - 0.22), nominal[1] - nominal[0], 0.44,
                               facecolor=colour, edgecolor=colour, zorder=3))
        ax.text((volume.t_start + volume.t_end) / 2, y + 0.40,
                f"[{volume.t_start:g}, {volume.t_end:g}) s", ha="center",
                fontsize=11, color=colour)
        ax.text(sum(nominal) / 2, y, name, ha="center", va="center", color="white", fontsize=10)
    y_claim = 4.05
    for tau in data["cg_b_resource_rows"]:
        ax.add_patch(Rectangle((4 * tau, y_claim - 0.3), 4, 0.6,
                               facecolor="#f8ddc3", edgecolor=ORANGE, lw=1.7, zorder=3))
        ax.text(4 * tau + 2, y_claim, f"τ={tau}", ha="center", va="center", fontsize=11, color=ORANGE)
    ax.text(40, y_claim + 0.48, "a = 1 for rows 8, 9, 10, 11; one visit gives four claims",
            ha="center", fontsize=11, color=ORANGE)
    for y, key, colour in (
        (2.35, "cg_b_conflicting_visit_steps", ORANGE),
        (0.95, "astar_b_blocked_entry_steps", BLUE),
    ):
        blocked = set(data[key])
        for step in range(6, 17):
            if step in blocked:
                ax.scatter(4 * step, y, s=145, marker="s", color=colour, zorder=4)
            else:
                ax.scatter(4 * step, y, s=98, marker="o", facecolors="white",
                           edgecolors=GREEN, linewidths=1.8, zorder=4)
    ax.text(40, 2.86, "Another visit conflicts when its own four rows overlap B's rows",
            ha="center", fontsize=11, color=ORANGE)
    ax.annotate(
        "Extra blocked step at 56 s\nin A*'s conservative raster",
        xy=(56, 0.95), xytext=(51.5, 1.62), ha="center", fontsize=10.5, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2),
    )
    ax.set_yticks([*y_ledger, y_claim, 2.35, 0.95])
    ax.set_yticklabels([
        "Ledger: incoming box\nA* and CG",
        "Ledger: outgoing box\nA* and CG",
        "CG: resource periods\nclaimed by source flight",
        "CG: conflicting visit times\nof another flight at B",
        "A*: blocked entry times\nof another flight at B",
    ], fontsize=12)
    ax.tick_params(axis="y", length=0, pad=15)
    ax.tick_params(axis="x", labelsize=12)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b6c3cc")
    for y in (4.95, 3.18):
        ax.axhline(y, color="#e0e7ec", lw=1.0)
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="Unbuffered movement"),
            Patch(facecolor="#c7ddec", edgecolor=BLUE, label="Buffered ledger window"),
            Line2D([], [], color=GREY, marker="s", ls="", markersize=9, label="Blocked candidate step"),
            Line2D([], [], markeredgecolor=GREEN, markerfacecolor="white", marker="o", ls="",
                   markersize=9, label="Unblocked at B"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.115), ncol=4, frameon=False, fontsize=11,
    )
    fig.text(0.5, 0.075,
             "CG's 4 occupied rows imply 7 conflicting candidate visit steps: {s−2, …, s+1} ∩ {8, 9, 10, 11} ≠ ∅.",
             ha="center", fontsize=12, color=DARK)
    fig.text(0.5, 0.038,
             "This compares cell B for the two-hop excerpt only. A full candidate must also satisfy every other cell and endpoint constraint.",
             ha="center", fontsize=10.5, color=GREY)
    fig.savefig(OUT / "temporal.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": DARK, "axes.labelcolor": GREY, "axes.titlecolor": DARK,
        "xtick.color": GREY, "ytick.color": DARK,
    })
    cfg, volumes, cells, blocked, claimed, data = construct()
    spatial(cfg, volumes, cells, blocked, claimed)
    temporal(cfg, volumes, data)
    (OUT / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(dict(out=str(OUT), **data), indent=2))


if __name__ == "__main__":
    main()
