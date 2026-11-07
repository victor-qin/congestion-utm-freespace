"""Registry and demand calibration for the column-generation acceptance world."""

import json

import numpy as np

from experiments.run import build_parser
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.scenarios import SCENARIOS, ScenarioSpec, get_scenario
from freespace_sim.scenarios.colgen import COLGEN_USS


def test_colgen_test_registered_with_density_miniature_parameters():
    spec = get_scenario("colgen_test")
    assert spec is SCENARIOS["colgen_test"]
    assert spec.name == "colgen_test"
    assert not spec.name.startswith("density_")
    assert spec.description

    cfg = spec.config()
    assert cfg.region_size_m == (8_000.0, 8_000.0)
    assert cfg.horizon_s == 1_800.0
    assert cfg.effective_demand_duration_s == 300.0
    assert cfg.lam_per_hour == 600.0
    assert cfg.seed == 0
    assert cfg.fixed_exit_lanes is True
    assert cfg.terminal_airspace_always_active is True
    assert cfg.flight_levels_m == (100.0,)

    demand = spec.demand_model()
    assert isinstance(demand, HubRadiusDemand)
    assert demand.n_hubs_per_uss == {COLGEN_USS: 8}
    assert demand.radius_m == {COLGEN_USS: 2_500.0}
    assert demand.pads_per_hub == {COLGEN_USS: 8}
    assert demand.terminal_radius_m == {COLGEN_USS: 180.0}
    assert demand.lam_per_uss == {COLGEN_USS: 600.0}
    assert demand.departure_offset_s == {COLGEN_USS: (120.0, 30.0)}
    # ONE-WAY: `run_batch` refuses a round-trip itinerary, so this scenario cannot use it.
    assert demand.return_flights is False
    assert demand.turnaround_s is None          # inherits cfg.turnaround_s
    assert demand.timing_mode == "departure"


def test_colgen_test_seed_zero_generates_calibrated_one_way_load():
    spec = get_scenario("colgen_test")
    cfg = spec.config()
    demand = spec.demand_model()
    requests = demand.generate(cfg, np.random.default_rng(cfg.seed))

    # 600 deliveries/hour over five minutes has expectation 50; the pinned seed realizes 49. One-way,
    # so the load is HALF the pre-itinerary scenario's and its numbers are not comparable.
    assert len(requests) == 49
    assert not any(r.return_to_origin for r in requests)
    assert all(r.origin_terminal is not None and r.dest_terminal is None for r in requests)
    assert {r.uss_id for r in requests} == {COLGEN_USS}
    assert {r.origin_terminal.id for r in requests} == {f"{COLGEN_USS}#{i}" for i in range(8)}

    # None on a one-way flight: there is no pad dwell for it to describe.
    assert all(r.turnaround_s is None for r in requests)


def test_colgen_test_spec_round_trips_and_run_parser_accepts_it():
    spec = get_scenario("colgen_test")
    payload = json.loads(json.dumps(spec.to_json_dict()))
    assert ScenarioSpec.from_json_dict(payload) == spec

    args = build_parser().parse_args(["--scenario", "colgen_test", "--planner", "colgen"])
    assert (args.scenario, args.planner) == ("colgen_test", "colgen")
