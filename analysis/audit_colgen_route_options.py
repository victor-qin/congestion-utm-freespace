"""Measure priced route length and ground delay relative to each original seed."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.types import Terminal
from freespace_sim.volumes import enroute_flown_m, enroute_reference_m


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(case):
    inputs = read(case / "inputs.json")
    cfg = SimConfig(**inputs["config"])
    requests = {r["flight_id"]: r for r in inputs["requests"]}
    columns, paths = jsonl(case / "columns.jsonl"), jsonl(case / "paths.jsonl")
    seeds = {}
    for c in columns:
        seeds.setdefault(c["flight_id"], c)
    spacing = cfg.dt_s * cfg.nominal_speed_mps
    route_metrics = []
    for path in paths:
        request = requests[path["flight_id"]]
        cells = np.array(path["cells"], dtype=float)
        xy = spacing * np.column_stack((cells[:, 0] + cells[:, 1] / 2,
                                        math.sqrt(3) / 2 * cells[:, 1]))
        xyz = np.column_stack((xy, np.full(len(xy), cfg.flight_levels_m[path["level"]])))
        terms = [Terminal(*request[key]) if request[key] is not None else None
                 for key in ("origin_terminal", "dest_terminal")]
        reference = enroute_reference_m(request["origin"], request["dest"], *terms, cfg)
        flown = enroute_flown_m(xyz, request["origin"], request["dest"], *terms, cfg)
        holds_s = cfg.dt_s * sum(a == b for a, b in zip(path["cells"], path["cells"][1:]))
        assert reference > 0
        route_metrics.append(dict(flown_m=flown, reference_m=reference, holds_s=holds_s,
                                  cells=len(cells), points=xy))
    counts, records = Counter(), []
    for c in columns:
        seed = seeds[c["flight_id"]]
        route, seed_route = route_metrics[c["path_id"]], route_metrics[seed["path_id"]]
        base = math.ceil(requests[c["flight_id"]]["t_departure"] / cfg.dt_s)
        assert seed["departure_step"] == base
        ground_s = (c["departure_step"] - base) * cfg.dt_s
        expected_cost = (cfg.cost_ground_delay_per_s * ground_s
                         + cfg.cost_air_hold_per_s * route["holds_s"]
                         + cfg.cost_air_lateral_per_s
                         * max(0, route["flown_m"] - route["reference_m"]) / cfg.nominal_speed_mps)
        assert abs(expected_cost - c["cost"]) < 1e-5, (c["index"], expected_cost, c["cost"])
        if c["phase"] != "pricing":
            continue
        delta_m = route["flown_m"] - seed_route["flown_m"]
        assert delta_m >= -1e-4, "A priced route is shorter than its deterministic shortest seed"
        same_geometry = c["geometry_id"] == seed["geometry_id"]
        same_length = abs(delta_m) < 1e-4
        kind = ("Original seed geometry" if same_geometry else
                "Different geometry, same length" if same_length else "Longer route")
        timing = "No ground delay" if ground_s == 0 else "Ground delayed"
        counts[(kind, timing)] += 1
        records.append(dict(column=c["index"], flight=c["flight_id"], sweep=c["sweep"],
                            seed_column=seed["index"], kind=kind, timing=timing,
                            ground_s=ground_s, holds_s=route["holds_s"],
                            flown_m=route["flown_m"], seed_flown_m=seed_route["flown_m"],
                            reference_m=route["reference_m"], extra_m_vs_seed=delta_m))
    summary = dict(case=str(case), columns=len(columns), priced_columns=len(records),
                   initialization_columns=sum(c["phase"] == "initialization" for c in columns),
                   counts={" / ".join(k): v for k, v in counts.items()},
                   no_ground_delay=sum(r["ground_s"] == 0 for r in records),
                   ground_delayed=sum(r["ground_s"] > 0 for r in records),
                   holds=sum(r["holds_s"] > 0 for r in records),
                   new_spatial_families=sum(c["new_geometry"] for c in columns if c["phase"] == "pricing"),
                   route_length_tolerance_m=1e-4, route_costs_verified=True,
                   definition="Flown horizontal distance after terminal folding and customer connectors; "
                              "compare to the initial deterministic shortest seed in the discrete graph")
    return summary, records, columns, route_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, records, columns, paths = audit(args.case.resolve())
    (args.out / "route_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "priced_route_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    candidates = [r for r in records if r["kind"] == "Different geometry, same length"
                  and r["ground_s"] == 0 and r["holds_s"] == 0]
    selected = max(candidates, key=lambda r: np.linalg.norm(
        paths[columns[r["column"]]["path_id"]]["points"].mean(axis=0)
        - paths[columns[r["seed_column"]]["path_id"]]["points"].mean(axis=0)))
    (args.out / "route_example.json").write_text(json.dumps(selected, indent=2) + "\n")
    seed = paths[columns[selected["seed_column"]]["path_id"]]
    alternative = paths[columns[selected["column"]]["path_id"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for path, label, color in ((seed, "Initial shortest seed", "#596F91"),
                               (alternative, "New priced route", "#07857C")):
        xy = path["points"] / 1000
        axes[0].plot(xy[:, 0], xy[:, 1], label=label, color=color, linewidth=2)
    axes[0].set(aspect="equal", xlabel="East (km)", ylabel="North (km)",
                title=f"Flight {selected['flight']}: different paths, equal flown distance")
    axes[0].legend(fontsize=9)
    axes[0].text(.03, .03, f"Both: {selected['flown_m']:,.1f} m · zero ground delay\n"
                 f"Priced column {selected['column']} added in sweep {selected['sweep']}",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor="white", alpha=.95, edgecolor="none"))
    groups = ["Different geometry, same length", "Longer route", "Original seed geometry"]
    labels = ["Different route,\nsame length", "Longer route", "Original route"]
    left = np.zeros(3)
    for timing, color in (("No ground delay", "#07857C"), ("Ground delayed", "#C78332")):
        values = [summary["counts"].get(f"{kind} / {timing}", 0) for kind in groups]
        axes[1].barh(labels, values, left=left, label=timing, color=color)
        left += values
    for y, value in enumerate(left):
        axes[1].text(value + 20, y, f"{int(value):,}", va="center")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Columns added by pricing", title="Actual outbound-only column mix")
    axes[1].set_xlim(0, max(left) * 1.15)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Pricing adds equal-length route alternatives as well as timing choices", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.out / "route_options.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
