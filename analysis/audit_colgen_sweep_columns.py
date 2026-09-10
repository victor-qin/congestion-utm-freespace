"""Independently recertify every first-sweep column and verify shortcut flights."""
import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
import sys
import time


def main():
    """Audit fresh-graph claims, translated costs, RCs and reference route differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.source_root.resolve()))
    from analysis.benchmark_colgen_lns import dump
    from analysis.replay_colgen_first_sweep import reference_columns, identity
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph, column_claims
    from freespace_sim.planner.colgen import pricing
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.translate import Column, column_to_intent

    assert args.source_root.resolve() in Path(pricing.__file__).resolve().parents
    with args.capture.open("rb") as handle:
        cap = pickle.load(handle)
    with (args.sweep / "columns.pkl").open("rb") as handle:
        columns = pickle.load(handle)
    data = json.loads((args.sweep / "sweep.json").read_text())
    rcs = dict(zip(data["flight_ids"], data["reduced_costs"], strict=True))
    cfg, params = cap["cfg"], cap["params"]
    catalog = StaticTerminalCatalog(cap["static_terminals"], cfg)
    view = pricing.DualView(cap["duals"], cfg)
    model = cost_model(cfg, params)
    requests = {r.flight_id: r for r in cap["requests"]}
    _, reference = reference_columns(args.reference, cap["first_lp"]["lp_column_count"])
    reference_by_flight = {key[0]: (key, cost) for key, cost in reference.items()}
    with gzip.open(args.reference / "flight_records/round_0001.jsonl.gz", "rt") as handle:
        reference_records = {r["flight_id"]: r for r in map(json.loads, handle)}
    failures, changes, early = [], [], []
    started = time.perf_counter()
    checked = 0
    for fid, column in columns.items():
        if column is None:
            continue
        graph = build_flight_graph(requests[fid], cfg, catalog, params)
        try:
            intent = column_to_intent(column, requests[fid], cfg)
            claims = column_claims(column, graph, cfg)
            cost = model.intent_cost(intent, cfg)
            rc = model.reduced_cost(benefit=pricing._benefit(params), cost=cost,
                                   dual_cost=view.claim_cost(claims), pi_f=cap["flight_duals"][fid])
            if not intent.accepted or claims != column.claims or not math.isclose(cost, column.delay_s, abs_tol=1e-7, rel_tol=0) or not math.isclose(rc, rcs[fid], abs_tol=1e-6, rel_tol=0):
                failures.append(dict(flight_id=fid, accepted=intent.accepted,
                    claims_match=claims == column.claims, stored_cost=column.delay_s, actual_cost=cost,
                    reported_rc=rcs[fid], actual_rc=rc))
            checked += 1
            old = reference_by_flight.get(fid)
            if old is not None and identity(column) != old[0]:
                key, old_cost = old
                old_column = Column(*key[:5], key[5], old_cost)
                cold = build_flight_graph(requests[fid], cfg, catalog, params)
                old_claims = column_claims(old_column, cold, cfg)
                old_actual_cost = model.intent_cost(column_to_intent(old_column, requests[fid], cfg), cfg)
                old_rc = model.reduced_cost(benefit=pricing._benefit(params), cost=old_actual_cost,
                    dual_cost=view.claim_cost(old_claims), pi_f=cap["flight_duals"][fid])
                changes.append(dict(flight_id=fid, old_cost=old_actual_cost, new_cost=cost,
                    cost_change=cost-old_actual_cost, old_rc=old_rc, new_rc=rc,
                    rc_equal=math.isclose(old_rc, rc, rel_tol=0, abs_tol=1e-6),
                    old_claim_count=len(old_claims), new_claim_count=len(claims),
                    claims_removed=len(old_claims-claims), claims_added=len(claims-old_claims),
                    old_dual_cost=view.claim_cost(old_claims), new_dual_cost=view.claim_cost(claims)))
        except Exception as exc:
            failures.append(dict(flight_id=fid, error=repr(exc)))
        if checked % 200 == 0:
            print(f"Recertified {checked} returned columns", flush=True)
    # Missing final_rc identifies the seed-optimal shortcut in the old capture. Rebuild
    # its global minimum-cost certificate explicitly without any bootstrap or DP call.
    for fid, record in reference_records.items():
        if "final_rc" in record:
            continue
        graph = build_flight_graph(requests[fid], cfg, catalog, params)
        seed = pricing.seed_column(graph, cfg, model=model)
        seed_cost = view.claim_cost(seed.claims)
        expected = model.reduced_cost(benefit=pricing._benefit(params), cost=seed.delay_s,
                                     dual_cost=seed_cost, pi_f=cap["flight_duals"][fid])
        certified = (graph._search_cache.seed_delay_certified and seed_cost == 0
                     and view.max_negative_credit == 0)
        matches = fid in rcs and math.isclose(expected, rcs[fid], abs_tol=1e-6, rel_tol=0)
        early.append(dict(flight_id=fid, certified=certified, expected_rc=expected,
                          reported_rc=rcs.get(fid), matches=matches))
    result = dict(returned_columns=sum(c is not None for c in columns.values()),
        recertified_columns=checked, failures=failures, changed_path_columns=len(changes),
        changed_paths=changes, changed_paths_same_rc=all(c["rc_equal"] for c in changes),
        changed_paths_same_cost=all(abs(c["cost_change"]) <= 1e-7 for c in changes),
        changed_claim_sets=sum(bool(c["claims_added"] or c["claims_removed"]) for c in changes),
        early_exit_flights=len(early), early_exit_checks=early,
        early_exit_all_certified=all(c["certified"] and c["matches"] for c in early),
        elapsed_s=time.perf_counter()-started, source_root=str(args.source_root.resolve()),
        note="Each returned column checked against fresh graph; reference alternatives use a separate fresh graph. Identity parity remains a separate honest failure for tied optima.")
    result["passed"] = bool(not failures and result["changed_paths_same_rc"] and result["early_exit_all_certified"])
    dump(args.sweep / "correctness.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"changed_paths", "early_exit_checks"}}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
