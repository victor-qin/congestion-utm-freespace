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
from .types import FlightRequest, Terminal, vec


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


_MAX_HUB_ATTEMPTS = 20000


def _scatter_hubs(cfg, rng, n_hubs_per_uss, radius_of, gap_m):
    """Uniform-scatter hub centres, reject-sampled so **no two terminal airspaces overlap**.

    Every accepted centre keeps a distance of at least ``r_i + r_j + gap_m`` to every other hub, where
    ``r`` is the hub's terminal (column) radius and ``gap_m`` is the clearance left between airspace
    *edges* — enough for an approach corridor to fit between neighbours. Without this, an unconstrained
    ``rng.uniform`` scatter occasionally drops two same-operator hubs within a radius of each other, and
    under ``terminal_airspace_always_active`` one hub's permanent wall then engulfs the other's landing
    approach, making its flights near-infeasible (the walls are transient without the flag, so the
    overlap is a latent modelling wart there rather than a hard failure).

    Deterministic in ``rng``; placement depends only on the region, hub counts and radii (not pad
    capacity or the demand seed). Raises ``ValueError`` if the region is too crowded to satisfy the
    separation — a mis-specified scenario fails loudly instead of silently overlapping."""
    w, h = cfg.region_size_m
    xs: list[float] = []
    ys: list[float] = []
    rs: list[float] = []
    out: dict[str, np.ndarray] = {}
    for uid, k in n_hubs_per_uss.items():
        r = float(radius_of(uid))
        pts = np.empty((k, 2), float)
        for i in range(k):
            for _ in range(_MAX_HUB_ATTEMPTS):
                c = rng.uniform([0.0, 0.0], [w, h])
                ok = True
                for j in range(len(xs)):
                    need = r + rs[j] + gap_m
                    if (c[0] - xs[j]) ** 2 + (c[1] - ys[j]) ** 2 < need * need:
                        ok = False
                        break
                if ok:
                    xs.append(float(c[0]))
                    ys.append(float(c[1]))
                    rs.append(r)
                    pts[i] = c
                    break
            else:
                raise ValueError(
                    f"place_hubs: could not position a '{uid}' hub with a {gap_m:.0f} m edge gap after "
                    f"{_MAX_HUB_ATTEMPTS} attempts — region {w:.0f}x{h:.0f} m is too crowded for "
                    f"{sum(n_hubs_per_uss.values())} hubs at these terminal radii "
                    f"(lower min_hub_gap_m, shrink terminal radii, or enlarge the region)."
                )
        out[uid] = pts
    return out


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
        process (town-centre seeds + Gaussian spread) to mimic real retail geography. Unlike
        :class:`HubRadiusDemand`, these flights carry no ``origin_terminal``/``dest_terminal`` (see
        ``generate`` below), so there are no terminal airspaces to overlap — hence no
        minimum-separation reject-sampling here (that belongs only where hubs build walls).
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


def _sample_in_disk(center: np.ndarray, radius_m: float, rng: np.random.Generator) -> np.ndarray:
    """A point drawn uniformly in the disk of radius ``radius_m`` about ``center`` (area-uniform)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r = radius_m * np.sqrt(rng.uniform(0.0, 1.0))
    return np.asarray(center, float) + r * np.array([np.cos(theta), np.sin(theta)])


@dataclass
class HubRadiusDemand:
    """Hub-and-spoke demand for a realistic metro vertiport study — three differences from
    :class:`HubVoronoiDemand`, each a knob the bottleneck analysis asked for:

    - **multi-pad hubs** (``pads_per_hub``): each hub is a *single location* that is a shared
      vertiport terminal with capacity N — up to N flights take off/land concurrently, the (N+1)th
      takes ground delay. Modelled via ``FlightRequest.origin_terminal``/``dest_terminal`` =
      ``(hub_id, N)``; the planner shares the hub's terminal column among its own flights (see
      ``conflict.volumes_conflict``) and bounds concurrency at N (occupancy). No spatial pad-spreading.
    - **radius service areas** (``radius_m``, ``float`` or per-USS ``dict``): a customer is drawn
      uniformly in the *disk* of that radius about a hub. Overlapping disks create crossing traffic
      and bound flight length directly.
    - **return flights** (``return_flights``): each delivery (hub → customer) is followed by a return
      (customer → the *same hub*, landing on any open pad), filed at the delivery's estimated arrival
      + ``turnaround_s``. The return's landing also consumes a pad, counted against the hub's N.

    ``lam_per_hour`` counts *deliveries*; with returns on, the realised flight count is ~2×. Hubs are
    placed once under ``hub_seed`` (stable infrastructure); only demand varies with ``cfg.seed``.
    """

    n_hubs_per_uss: dict[str, int] = field(
        default_factory=lambda: {"walmart_uss": 6, "stripmall_uss": 20}
    )
    radius_m: "float | dict[str, float]" = 3000.0   # customer demand radius (scalar, or per-USS)
    pads_per_hub: int = 1                            # terminal capacity N per hub
    terminal_radius_m: "float | dict[str, float] | None" = None   # column size; None → hover footprint
    corridor_overlap_m: "float | None" = None        # exit-lane overlap into column; None/0 → flush at edge
    return_flights: bool = True                      # each delivery → a return to its origin hub
    turnaround_s: float = 0.0                      # delay before the return is filed (after est. arrival)
    uss_share: dict[str, float] | None = None
    min_od_separation_m: float = 200.0
    hub_seed: int = 0xA17F
    min_hub_gap_m: float = 100.0                     # clearance between terminal-airspace EDGES (no overlap)

    def place_hubs(self, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return ``{uss_id: (n_hubs, 2)}`` single-point hub centres, reject-sampled so no two hubs'
        terminal airspaces overlap (:func:`_scatter_hubs`; each USS's column radius is
        ``terminal_radius_m`` or the ``cfg`` hover footprint). DESIGN KNOB for spatial structure (swap
        the uniform scatter for a clustered process to mimic real retail geography)."""
        def radius_of(uid: str) -> float:
            tr = self._terminal_radius_for(uid)
            return cfg.terminal_radius_m if tr is None else float(tr)
        return _scatter_hubs(cfg, rng, self.n_hubs_per_uss, radius_of, self.min_hub_gap_m)

    def _radius_for(self, uss_id: str) -> float:
        return float(self.radius_m[uss_id] if isinstance(self.radius_m, dict) else self.radius_m)

    def _terminal_radius_for(self, uss_id: str) -> float | None:
        tr = self.terminal_radius_m
        if tr is None:
            return None                              # builder defaults to the hover footprint
        return float(tr[uss_id] if isinstance(tr, dict) else tr)

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
        hubs = self.place_hubs(cfg, np.random.default_rng(self.hub_seed))
        ids, probs = self._shares()
        n = int(rng.poisson(cfg.lam_per_hour * cfg.horizon_s / 3600.0))

        requests: list[FlightRequest] = []
        fid = 0
        for _ in range(n):
            uss_id = ids[int(rng.choice(len(ids), p=probs))]
            hi = int(rng.integers(hubs[uss_id].shape[0]))
            hub = hubs[uss_id][hi]
            terminal = Terminal(f"{uss_id}#{hi}", int(self.pads_per_hub),
                                self._terminal_radius_for(uss_id), self.corridor_overlap_m)
            radius = self._radius_for(uss_id)
            customer = None
            for _ in range(20):  # redraw until in-region and a non-trivial hop from the hub
                c = _sample_in_disk(hub, radius, rng)
                if 0.0 <= c[0] <= w and 0.0 <= c[1] <= h and \
                        np.linalg.norm(c - hub) >= self.min_od_separation_m:
                    customer = c
                    break
            if customer is None:
                customer = np.clip(c, [0.0, 0.0], [w, h])
            t_req = float(rng.uniform(0, cfg.horizon_s))
            requests.append(FlightRequest(                            # delivery: hub → customer
                fid, vec(hub[0], hub[1], gl), vec(customer[0], customer[1], gl), t_req,
                uss_id=uss_id, origin_terminal=terminal))
            fid += 1
            if self.return_flights:                                  # return: customer → same hub
                t_ret = t_req + self._est_trip_s(hub, customer, cfg) + self.turnaround_s
                requests.append(FlightRequest(
                    fid, vec(customer[0], customer[1], gl), vec(hub[0], hub[1], gl), t_ret,
                    uss_id=uss_id, dest_terminal=terminal))
                fid += 1

        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests
