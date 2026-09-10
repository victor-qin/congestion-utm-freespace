"""Illustrate certificate reuse with actual geometry and independently checked CG rows."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

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

FIGURES = ROOT / "context/figures"
OUT = ROOT / "analysis/colgen_certificate_reuse_20260910"
BLUE, ORANGE, GREEN, INK, GRAY = "#1767A6", "#CE6B22", "#217B66", "#172F43", "#617281"


def box_polygon(spec):
    """Project a real ledger box onto the horizontal plane."""
    corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    local = corners * np.asarray(spec.extents) / 2
    return (local @ spec.rotation().T + np.asarray(spec.center))[:, :2]


def construct():
    """Build one route, shift its cached rows, and compare with a cold reconstruction."""
    source = ROOT / ("analysis/colgen_todos_hybrid_1200s_20260910/combined/attempt_01/"
                     "density_faa_wing_zipline_seed0_iteration_ip_eager/inputs.json")
    raw = json.loads(source.read_text())["config"]
    cfg = SimConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    radius = hg.circumradius(cfg)
    path = ((-3, 0), (-2, 0), (-1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1))

    def point(cell):
        return np.r_[hg.hex_center(*cell, radius), cfg.ground_level_m]

    request = FlightRequest(9001, point(path[0]), point(path[-1]), 0.0)
    wall = Terminal("permanent", 1, radius=90.0)
    terms = [(point((0, 3)), wall)]
    params = ColGenParams()
    graph = build_flight_graph(request, cfg, terms, params)
    base = translate.Column(request.flight_id, 2, 0, None, None, path, 0.0)
    intent = translate.column_to_intent(base, request, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    model = cost_model(cfg, params)
    base = replace(base, delay_s=model.intent_cost(intent, cfg))
    original = translate.column_to_intent
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0].departure_step)
        return original(*args, **kwargs)

    translate.column_to_intent = observed
    try:
        claims = column_claims(base, graph, cfg)
        shifted = replace(base, departure_step=5,
                          delay_s=base.delay_s + model.ground_weight * 3 * cfg.dt_s)
        shifted_claims = column_claims(shifted, graph, cfg)
    finally:
        translate.column_to_intent = original
    assert calls == [base.departure_step], calls
    delta = shifted.departure_step - base.departure_step
    expected = frozenset(
        RowKey.cell(*r.cell_coord, r.level, r.step + delta) if r.kind == "cell"
        else RowKey.term(r.terminal_id, r.step + delta) for r in claims)
    cold = column_claims(shifted, build_flight_graph(request, cfg, terms, params), cfg)
    assert shifted_claims == expected == cold
    shifted_intent = original(shifted, request, cfg)
    assert shifted_intent.status is IntentStatus.ACCEPTED
    assert abs(model.intent_cost(shifted_intent, cfg) - shifted.delay_s) < 1e-8
    assert len(intent.volumes) == len(shifted_intent.volumes)
    for before, after in zip(intent.volumes, shifted_intent.volumes, strict=True):
        assert before.shape == after.shape
        assert abs(after.t_start - before.t_start - delta * cfg.dt_s) < 1e-8
        assert abs(after.t_end - before.t_end - delta * cfg.dt_s) < 1e-8
    arrival = base.departure_step + graph.takeoff_steps[0] + len(path) - 1
    windows = _column_endpoint_steps(base, graph, cfg, arrival)
    offsets = derive_cell_window(cfg)
    selected = path[2:5]
    rows = {name: {str(cell): sorted(r.step for r in collection
              if r.kind == "cell" and r.cell_coord == cell and r.level == 0)
              for cell in selected}
            for name, collection in (("base", claims), ("shifted", shifted_claims))}
    assert rows["base"][str((0, 0))] == [8, 9, 10, 11]
    assert rows["shifted"][str((0, 0))] == [11, 12, 13, 14]
    data = dict(config_source=str(source), dt_s=cfg.dt_s, path=path, departure_steps=[2, 5],
                shift_steps=delta, departure_seconds=[2 * cfg.dt_s, 5 * cfg.dt_s],
                cell_window_offsets=offsets, takeoff_steps=graph.takeoff_steps[0],
                claim_count=len(claims), shifted_claim_count=len(shifted_claims),
                expensive_translations_for_base_and_cached_shift=len(calls),
                cold_claims_equal=True, spatial_volume_shapes_equal=True,
                all_volume_times_shift_by_s=delta * cfg.dt_s,
                relative_endpoint_windows=[(w.start - base.departure_step,
                                            w.stop - base.departure_step) for w in windows],
                base_cost=base.delay_s, shifted_cost=shifted.delay_s, shown_rows=rows,
                scope="Illustrative route under benchmark configuration, not a benchmark flight.")
    return cfg, graph, base, intent, terms, data


def panel_box(ax, xy, width, height, title, lines, color):
    """Draw a readable cache/validation panel in normalized coordinates."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015",
                              facecolor=color, edgecolor="#D3DFE5", linewidth=1))
    ax.text(x + .025, y + height - .055, title, va="top", weight="bold", fontsize=15, color=INK)
    ax.text(x + .025, y + height - .118, lines, va="top", fontsize=12.5,
            linespacing=1.65, color=INK)


def spatial(cfg, graph, base, intent, terms, data):
    """Show the geometry that is validated and the precise data retained in the cache."""
    fig = plt.figure(figsize=(15, 7.7), facecolor="white")
    fig.suptitle("Same spatial route: reuse the validation result", fontsize=22,
                 weight="bold", color=INK, y=.975)
    fig.text(.5, .918, "Departure 8 s → 20 s changes the clock; the flight's spatial shape is unchanged",
             ha="center", fontsize=13.5, color=GRAY)
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
            ax.add_patch(Polygon(box_polygon(volume.shape), facecolor=BLUE,
                                 edgecolor=BLUE, alpha=.19, lw=1.4))
    for center, terminal in terms:
        ax.add_patch(Circle(center[:2], terminal.radius, facecolor="#F9DDDB",
                            edgecolor="#AD514E", linestyle="--", lw=2))
        ax.text(center[0], center[1], "Permanent\nairspace", ha="center", va="center",
                fontsize=12, color="#85403F")
    xy = np.array([hg.hex_center(*c, radius) for c in base.cell_path])
    ax.plot(*xy.T, color=BLUE, lw=3, zorder=4)
    ax.scatter(*xy.T, color=BLUE, s=28, zorder=5)
    for name, cell in zip("ABC", base.cell_path[2:5], strict=True):
        pt = hg.hex_center(*cell, radius)
        ax.annotate(name, pt, xytext=(0, -28), textcoords="offset points", ha="center",
                    weight="bold", fontsize=14, color=INK)
    for label, point in (("Origin", xy[0]), ("Destination", xy[-1])):
        ax.add_patch(Circle(point, cfg.effective_hover_radius_m, facecolor=BLUE, alpha=.15))
        ax.annotate(label, point, xytext=(0, -43), textcoords="offset points",
                    ha="center", fontsize=12, color=INK)
    ax.set(xlim=(-470, 550), ylim=(-170, 475), aspect="equal")
    ax.set_axis_off()
    ax.text(.5, -.10, "Shaded polygons are actual filed corridor boxes.\n"
            "The same path, altitude, and lane choices identify the cached route.",
            ha="center", transform=ax.transAxes, fontsize=12, color=GRAY, linespacing=1.5)
    right = fig.add_axes((.56, .13, .40, .74))
    right.set(xlim=(0, 1), ylim=(0, 1))
    right.set_axis_off()
    panel_box(right, (.025, .69), .94, .28, "First departure: validate once",
              "Check cells, hops, endpoint / lane agreement,\n"
              "translated geometry, detour limits, static walls.\n"
              f"Build the union of {data['claim_count']} capacity-row IDs.", "#EAF2F9")
    panel_box(right, (.025, .36), .94, .25, "Save a successful cache entry",
              r"$K=(f, R, \ell, L_o, L_d)$" + "  (within this graph / configuration)\n"
              r"$K\ \mapsto\ (d_0, A_0, \omega_o, \omega_d)$" + "\n"
              "Base departure, row IDs, relative endpoint windows.", "#E9F4EF")
    panel_box(right, (.025, .025), .94, .25, "Later departure: reuse + recheck",
              "Match K; check new departure / arrival bounds.\n"
              "Check the endpoint-window rounding guard.\n"
              "Shift row timestamps; retain the spatial validation.", "#FFF1E5")
    for top, bottom in ((.69, .61), (.36, .275)):
        right.annotate("", (.50, bottom + .01), (.50, top - .015),
                       arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.7))
    fig.text(.5, .025, "The master/IP still checks capacity conflicts with other flights for each timed alternative.",
             ha="center", color=INK, fontsize=13, weight="bold")
    fig.savefig(FIGURES / "colgen_certificate_spatial.png", dpi=160, facecolor="white")
    plt.close(fig)


def temporal(data):
    """Plot actual before/after capacity rows for three interior cells."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    fig.subplots_adjust(left=.10, right=.97, bottom=.21, top=.77, hspace=.65)
    fig.suptitle("Reuse the claim pattern by shifting its time indices", fontsize=22,
                 color=INK, weight="bold", y=.97)
    fig.text(.5, .898, r"$A_{d+\Delta}=T_\Delta(A_d)$" + "     •     Δ = 3 steps = 12 seconds",
             ha="center", fontsize=17, color=INK)
    fig.text(.5, .84, "One filled square = one capacity coefficient equal to 1; blank = 0",
             ha="center", fontsize=12.5, color=GRAY)
    cells = ((-1, 0), (0, 0), (0, 1))
    for ax, key, color, title in zip(axes, ("base", "shifted"), (BLUE, ORANGE),
            ("Original: departure step 2 (8 s)", "Alternative: departure step 5 (20 s)"), strict=True):
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
             ha="center", fontsize=17, color=INK)
    fig.text(.5, .035, "Shown: three interior cells. The verified full row set also includes endpoint occupancy.\n"
             "If endpoint rounding changes the relative windows, rows are rebuilt while spatial validation is reused.",
             ha="center", fontsize=12, color=GRAY, linespacing=1.5)
    fig.savefig(FIGURES / "colgen_certificate_temporal.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    """Write two PNGs and the numerical checks backing their content."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, graph, base, intent, terms, data = construct()
    (OUT / "verified_example.json").write_text(json.dumps(data, indent=2) + "\n")
    spatial(cfg, graph, base, intent, terms, data)
    temporal(data)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
