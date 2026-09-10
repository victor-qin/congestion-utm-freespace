"""Audit completed combined/wider6/multi3 pools without running any pricing."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace
from itertools import combinations
import json
import math
from pathlib import Path
import pickle
import sys


def load_case(case):
    """Read exact timed-path identities, costs, labels, and selected column origins."""
    with (case / "paths.jsonl").open() as handle:
        paths = {p["id"]: p for p in map(json.loads, handle)}
    with (case / "columns.jsonl").open() as handle:
        columns = list(map(json.loads, handle))
    selected = set(json.loads((case / "selections.json").read_text())["final"].values())
    timed, spatial, column_keys = {}, set(), {}
    for c in columns:
        p = paths[c["path_id"]]
        cells = tuple(tuple(cell) for cell in p["cells"])
        route = (c["flight_id"], p["level"], p["origin_lane_idx"], p["dest_lane_idx"], cells)
        key = (c["departure_step"],) + route
        if key in timed and timed[key] != c["cost"]:
            raise ValueError(f"One canonical timed path has different costs in {case}")
        timed[key] = c["cost"]
        column_keys[c["index"]] = key
        # Spatial identity removes consecutive holds but preserves the route's visit order.
        collapsed = tuple(cell for i, cell in enumerate(cells) if i == 0 or cells[i-1] != cell)
        spatial.add(route[:-1] + (collapsed,))
    labels = Counter()
    records = 0
    for file in sorted((case / "flight_records").glob("*.summary.json")):
        summary = json.loads(file.read_text())
        records += summary["records"]
        for key, value in summary.get("metrics", {}).items():
            if "labels" in key:
                labels[key] += value["total"]
    result = json.loads((case / "result.json").read_text())
    return dict(case=str(case), result=result, timed=timed, spatial=spatial, columns=columns,
                paths=paths, selected=selected, column_keys=column_keys, label_totals=dict(labels),
                records=records, rounds=len(json.loads((case / "iterations.json").read_text())),
                final_origins=dict(Counter(c["phase"] for c in columns if c["index"] in selected)))


def compare(a, b):
    """Compare exact identities, then distinguish exact and tolerance-level cost equality."""
    left, right = a["timed"], b["timed"]
    shared = left.keys() & right.keys()
    mismatch = [dict(identity=repr(k), left=left[k], right=right[k]) for k in shared
                if not math.isclose(left[k], right[k], rel_tol=0, abs_tol=1e-7)]
    return dict(left_columns=len(left), right_columns=len(right), shared_timed_paths=len(shared),
                left_only_timed_paths=len(left.keys()-right.keys()),
                right_only_timed_paths=len(right.keys()-left.keys()),
                identical_timed_pools=left.keys() == right.keys(),
                shared_exact_cost_equal=sum(left[k] == right[k] for k in shared),
                shared_tolerance_cost_equal=len(shared)-len(mismatch), cost_mismatches=mismatch,
                max_shared_cost_difference=max((abs(left[k]-right[k]) for k in shared), default=None),
                shared_spatial_paths=len(a["spatial"] & b["spatial"]),
                left_only_spatial_paths=len(a["spatial"]-b["spatial"]),
                right_only_spatial_paths=len(b["spatial"]-a["spatial"]))


def main():
    """After all three arms complete, inspect retained wider-domain use on fresh graphs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.runs / "status.json").read_text())
    names = ("combined", "wider6", "multi3")
    if any(status.get(name, {}).get("state") != "complete" for name in names):
        raise ValueError("Wait until combined, wider6 and multi3 have completed")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.source_root.resolve()))
    from analysis.benchmark_colgen_lns import dump
    from freespace_sim.planner.colgen import network
    from freespace_sim.planner.colgen.params import ColGenParams
    assert args.source_root.resolve() in Path(network.__file__).resolve().parents
    arms = {}
    for name in names:
        cases = list(Path(status[name]["output"]).glob("*_seed0_iteration_ip_eager"))
        if len(cases) != 1:
            raise ValueError(f"Expected one case for {name}")
        arms[name] = load_case(cases[0])
    if len({a["result"]["demand_config_sha"] for a in arms.values()}) != 1:
        raise ValueError("Demand/config differs across compared arms")
    # Only this trusted, locally generated pickle is loaded. No search is performed.
    with args.capture.open("rb") as handle:
        capture = pickle.load(handle)
    params = ColGenParams(**{f.name: getattr(capture["params"], f.name)
        for f in fields(ColGenParams) if hasattr(capture["params"], f.name)})
    params = replace(params, max_air_overrun_hops=3)
    cfg = capture["cfg"]
    catalog = network.StaticTerminalCatalog(capture["static_terminals"], cfg)
    requests = {r.flight_id:r for r in capture["requests"]}
    wide = arms["wider6"]
    caps = {}
    for fid in {c["flight_id"] for c in wide["columns"]}:
        caps[fid] = network.build_flight_graph(requests[fid], cfg, catalog, params).max_air_hops
    excess = []
    for c in wide["columns"]:
        hops = len(wide["paths"][c["path_id"]]["cells"])-1
        if hops > caps[c["flight_id"]]:
            excess.append(dict(column_index=c["index"], flight_id=c["flight_id"],
                phase=c["phase"], sweep=c["sweep"], hops=hops, standard_max_air_hops=caps[c["flight_id"]],
                excess_hops=hops-caps[c["flight_id"]], selected_final=c["index"] in wide["selected"]))
    results = dict(arms={name: {key:a[key] for key in (
        "case", "result", "label_totals", "records", "rounds", "final_origins")}
        | dict(pool_columns=len(a["columns"]), distinct_timed_paths=len(a["timed"]),
               distinct_spatial_paths=len(a["spatial"])) for name,a in arms.items()},
        pairwise={f"{a}_vs_{b}":compare(arms[a],arms[b]) for a,b in combinations(names,2)},
        wider_usage=dict(standard_overrun=3, wider_overrun=6,
            columns_exceeding_standard_cap=len(excess), flights_exceeding_standard_cap=len({c["flight_id"] for c in excess}),
            pricing_columns_exceeding_standard_cap=sum(c["phase"] == "pricing" for c in excess),
            final_selected_exceeding_standard_cap=sum(c["selected_final"] for c in excess), columns=excess),
        limitations=["Hop count includes repeated-cell holds, matching the airborne-step budget.",
            "Retained columns quantify actual pool/final use, not all paths explored during pricing.",
            "Claims and extra-versus-winning-column provenance are absent; neither is inferred.",
            "Runtime comparisons require cost, convergence gaps, termination and rounds; different tied routes can change later pools."],
        source_root=str(args.source_root.resolve()), capture=str(args.capture.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "domain_pool_audit.json", results)
    print(json.dumps({"wider_usage":{k:v for k,v in results["wider_usage"].items() if k != "columns"},
                      "pairwise":results["pairwise"]}), flush=True)


if __name__ == "__main__":
    main()
