"""Demand models — how flight requests are generated.

`UniformPoissonDemand`: Poisson(λ) arrivals over the horizon, origin/dest sampled uniformly in the
region at ground level, with a minimum O/D separation so requests are non-trivial. Deterministic
under a seeded RNG.

`HubVoronoiDemand`: same Poisson arrival process in *time*, but origins are geographically anchored —
each USS owns a fixed set of synthetic hubs and a flight runs from the *nearest* hub (its Voronoi
cell) to a random customer. Flights become short and convergent (cheap to plan, less denial) while
two overlapping hub tessellations keep crossing traffic high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import SimConfig
from .types import FlightRequest, vec


class DemandModel(Protocol):
    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]: ...


@dataclass
class UniformPoissonDemand:
    min_od_separation_m: float = 1000.0
    uss_ids: tuple[str, ...] = ("default",)

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        n = int(rng.poisson(cfg.lam_per_hour * cfg.horizon_s / 3600.0))
        requests: list[FlightRequest] = []
        for fid in range(n):
            for _ in range(20):  # rejection-sample until O/D are far enough apart
                o = rng.uniform([0, 0], [w, h])
                d = rng.uniform([0, 0], [w, h])
                if np.linalg.norm(d - o) >= self.min_od_separation_m:
                    break
            t_request = float(rng.uniform(0, cfg.horizon_s))
            uss_id = self.uss_ids[int(rng.integers(len(self.uss_ids)))]
            requests.append(
                FlightRequest(
                    flight_id=fid,
                    origin=vec(o[0], o[1], cfg.ground_level_m),
                    dest=vec(d[0], d[1], cfg.ground_level_m),
                    t_request=t_request,
                    uss_id=uss_id,
                )
            )
        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests


def nearest_hub(point: np.ndarray, hubs: np.ndarray) -> np.ndarray:
    """Return the row of ``hubs`` (shape ``(k, 2)``) closest to ``point`` — its Voronoi-cell owner."""
    return hubs[int(np.argmin(np.linalg.norm(hubs - point, axis=1)))]


@dataclass
class HubVoronoiDemand:
    """Hub-and-spoke demand: each USS serves a fixed set of synthetic ground hubs (think one USS for
    every *Walmart*, another for every *strip mall*). A customer is drawn uniformly and assigned a
    serving USS; the flight runs FROM that USS's *nearest* hub TO the customer (delivery). The flown
    length is bounded by the serving USS's Voronoi-cell radius — short, convergent, far cheaper than a
    uniform O/D dash across the whole metro — yet two USSs with *independent* hub tessellations cross
    each other's spokes and pile up on shared pads, so demand and conflict stay high.

    Arrivals are the *same* Poisson process in time as ``UniformPoissonDemand`` (count ``Poisson(λH)``,
    ``t_request ~ U(0, H)``); only the O/D *geometry* changes. Hubs are placed once under their own
    RNG (``hub_seed``) so the "infrastructure" is stable while only the demand varies with ``cfg.seed``
    — Walmarts don't move when you reroll traffic.
    """

    # hubs per USS — fewer hubs ⇒ bigger cells ⇒ longer flights (the two USSs differ on purpose)
    n_hubs_per_uss: dict[str, int] = field(
        default_factory=lambda: {"walmart_uss": 6, "stripmall_uss": 20}
    )
    uss_share: dict[str, float] | None = None       # demand split across USSs (None ⇒ equal)
    direction: str = "delivery"                     # "delivery" hub→customer | "pickup" customer→hub
    min_od_separation_m: float = 200.0              # reject trivially-short customer↔hub pairs
    hub_seed: int = 0xA17F                          # infrastructure RNG, independent of cfg.seed

    def place_hubs(self, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return ``{uss_id: (n_hubs, 2)}`` hub positions in region ENU metres.

        DESIGN KNOB — this is where the *spatial structure* of demand is decided. The default
        scatters hubs uniformly (already differentiating the USSs by density); swap in a clustered
        process (town-centre seeds + Gaussian spread) to mimic real retail geography.
        """
        w, h = cfg.region_size_m
        return {
            uid: rng.uniform([0.0, 0.0], [w, h], size=(k, 2))
            for uid, k in self.n_hubs_per_uss.items()
        }

    def _shares(self) -> tuple[list[str], np.ndarray]:
        ids = list(self.n_hubs_per_uss)
        if self.uss_share is None:
            p = np.ones(len(ids))
        else:
            p = np.array([self.uss_share.get(uid, 0.0) for uid in ids], float)
        return ids, p / p.sum()

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        hubs = self.place_hubs(cfg, np.random.default_rng(self.hub_seed))
        ids, probs = self._shares()
        n = int(rng.poisson(cfg.lam_per_hour * cfg.horizon_s / 3600.0))

        requests: list[FlightRequest] = []
        for fid in range(n):
            uss_id = ids[int(rng.choice(len(ids), p=probs))]
            uss_hubs = hubs[uss_id]
            for _ in range(20):  # redraw until the customer is a non-trivial hop from its hub
                customer = rng.uniform([0.0, 0.0], [w, h])
                hub = nearest_hub(customer, uss_hubs)
                if np.linalg.norm(customer - hub) >= self.min_od_separation_m:
                    break
            o, d = (hub, customer) if self.direction == "delivery" else (customer, hub)
            t_request = float(rng.uniform(0, cfg.horizon_s))
            requests.append(
                FlightRequest(
                    flight_id=fid,
                    origin=vec(o[0], o[1], cfg.ground_level_m),
                    dest=vec(d[0], d[1], cfg.ground_level_m),
                    t_request=t_request,
                    uss_id=uss_id,
                )
            )
        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests


def _pad_offsets(n_pads: int, spacing_m: float) -> np.ndarray:
    """``(n_pads, 2)`` pad offsets around a hub centre — a roughly-square grid, centred on (0, 0),
    spaced ``spacing_m`` apart so each pad's hover cylinder is independent of its neighbours'."""
    if n_pads <= 1:
        return np.zeros((1, 2))
    cols = int(np.ceil(np.sqrt(n_pads)))
    idx = np.arange(n_pads)
    rc = np.stack([idx % cols, idx // cols], axis=1).astype(float)   # (col, row)
    rc -= rc.mean(axis=0)                                            # centre on the hub
    return rc * spacing_m


def _sample_in_disk(center: np.ndarray, radius_m: float, rng: np.random.Generator) -> np.ndarray:
    """A point drawn uniformly in the disk of radius ``radius_m`` about ``center`` (area-uniform)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r = radius_m * np.sqrt(rng.uniform(0.0, 1.0))
    return np.asarray(center, float) + r * np.array([np.cos(theta), np.sin(theta)])


@dataclass
class HubRadiusDemand:
    """Hub-and-spoke demand for a realistic metro vertiport study — three differences from
    :class:`HubVoronoiDemand`, each a knob the bottleneck analysis asked for:

    - **multi-pad hubs** (``pads_per_hub``): each hub is *N* launch pads (spaced points), not one, so
      takeoffs parallelise. Pad throughput scales with pads, not just hubs — the binding constraint in
      the saturated runs. Pads are independent (spaced ≥ a hover diameter), so the existing simulator
      lets them launch concurrently with no engine change.
    - **radius service areas** (``radius_m``): a flight serves a customer drawn uniformly in the
      *disk* of radius ``radius_m`` about a hub, instead of the nearest-hub Voronoi cell. Overlapping
      disks create crossing traffic and bound flight length directly.
    - **return flights** (``return_flights``): each delivery (pad → customer) is followed by a return
      (customer → *the same pad*), filed at the delivery's estimated arrival + ``turnaround_s``.

    ``lam_per_hour`` counts *deliveries*; with returns on, the realised flight count is ~2×. Pads are
    placed once under ``hub_seed`` (stable infrastructure); only the demand varies with ``cfg.seed``.
    """

    n_hubs_per_uss: dict[str, int] = field(
        default_factory=lambda: {"walmart_uss": 6, "stripmall_uss": 20}
    )
    radius_m: float = 3000.0                # demand drawn within this radius of a hub
    pads_per_hub: int = 1                   # parallel launch pads per hub
    pad_spacing_m: float | None = None      # None ⇒ derived from the hover footprint (independent pads)
    return_flights: bool = True             # each delivery → a return to its origin pad
    turnaround_s: float = 120.0             # delay before the return is filed (after est. arrival)
    uss_share: dict[str, float] | None = None
    min_od_separation_m: float = 200.0
    hub_seed: int = 0xA17F

    def place_pads(self, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return ``{uss_id: (n_hubs, pads_per_hub, 2)}`` pad positions: hub centres scattered in the
        region, each expanded into a small grid of independent pads. DESIGN KNOB for spatial structure
        (swap the uniform scatter for a clustered process to mimic real retail geography)."""
        w, h = cfg.region_size_m
        spacing = self.pad_spacing_m if self.pad_spacing_m else 2.5 * cfg.effective_hover_radius_m
        offsets = _pad_offsets(self.pads_per_hub, spacing)              # (P, 2)
        pads = {}
        for uid, k in self.n_hubs_per_uss.items():
            centers = rng.uniform([0.0, 0.0], [w, h], size=(k, 2))      # (k, 2)
            pads[uid] = centers[:, None, :] + offsets[None, :, :]       # (k, P, 2)
        return pads

    def _shares(self) -> tuple[list[str], np.ndarray]:
        ids = list(self.n_hubs_per_uss)
        p = (np.ones(len(ids)) if self.uss_share is None
             else np.array([self.uss_share.get(uid, 0.0) for uid in ids], float))
        return ids, p / p.sum()

    def _est_trip_s(self, o: np.ndarray, d: np.ndarray, cfg: SimConfig) -> float:
        """Nominal door-to-door time for the return clock: cruise + climb/descent + one pad dwell."""
        dist = float(np.linalg.norm(np.asarray(d, float) - np.asarray(o, float)))
        return dist / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_s + cfg.hover_time_s

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        gl = cfg.ground_level_m
        pads = self.place_pads(cfg, np.random.default_rng(self.hub_seed))
        ids, probs = self._shares()
        n = int(rng.poisson(cfg.lam_per_hour * cfg.horizon_s / 3600.0))

        requests: list[FlightRequest] = []
        fid = 0
        for _ in range(n):
            uss_id = ids[int(rng.choice(len(ids), p=probs))]
            uss_pads = pads[uss_id]                                  # (k, P, 2)
            hi = int(rng.integers(uss_pads.shape[0]))
            pi = int(rng.integers(uss_pads.shape[1]))
            pad = uss_pads[hi, pi]
            center = uss_pads[hi].mean(axis=0)                       # hub centre (offsets sum to 0)
            customer = None
            for _ in range(20):  # redraw until in-region and a non-trivial hop from the pad
                c = _sample_in_disk(center, self.radius_m, rng)
                if 0.0 <= c[0] <= w and 0.0 <= c[1] <= h and \
                        np.linalg.norm(c - pad) >= self.min_od_separation_m:
                    customer = c
                    break
            if customer is None:
                customer = np.clip(c, [0.0, 0.0], [w, h])
            t_req = float(rng.uniform(0, cfg.horizon_s))
            requests.append(FlightRequest(
                fid, vec(pad[0], pad[1], gl), vec(customer[0], customer[1], gl), t_req, uss_id=uss_id))
            fid += 1
            if self.return_flights:                                 # customer → same pad, after dwell
                t_ret = t_req + self._est_trip_s(pad, customer, cfg) + self.turnaround_s
                requests.append(FlightRequest(
                    fid, vec(customer[0], customer[1], gl), vec(pad[0], pad[1], gl), t_ret, uss_id=uss_id))
                fid += 1

        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests
