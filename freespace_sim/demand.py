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

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import SimConfig
from .planner.hexgrid import SQRT3, circumradius, enu_to_axial, terminal_cells
from .volumes import exit_radius, terminal_radius
from .types import FlightRequest, Terminal, vec


class DemandModel(Protocol):
    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]: ...


@dataclass
class UniformPoissonDemand:
    min_od_separation_m: float = 1000.0
    uss_ids: tuple[str, ...] = ("default",)

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        demand_duration_s = cfg.effective_demand_duration_s
        n = int(rng.poisson(cfg.lam_per_hour * demand_duration_s / 3600.0))
        requests: list[FlightRequest] = []
        for fid in range(n):
            for _ in range(20):  # rejection-sample until O/D are far enough apart
                o = rng.uniform([0, 0], [w, h])
                d = rng.uniform([0, 0], [w, h])
                if np.linalg.norm(d - o) >= self.min_od_separation_m:
                    break
            t_request = float(rng.uniform(0, demand_duration_s))
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

    Every accepted centre keeps a distance of at least ``r_i + r_j + gap_m`` to every other hub —
    **across all operators**, since each candidate is checked against the one shared set of
    already-placed centres, not a per-USS one — where ``r`` is the hub's terminal (column) radius and
    ``gap_m`` is the clearance left between airspace *edges* (enough for an approach corridor to fit
    between neighbours). Without this, an unconstrained ``rng.uniform`` scatter occasionally drops two
    hubs within a radius of each other (the observed engulfment happened to be same-operator —
    ``stripmall_uss#11``/``#17`` — but the check spans operators), and under
    ``terminal_airspace_always_active`` one hub's permanent wall then engulfs the other's landing
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
        demand_duration_s = cfg.effective_demand_duration_s
        hubs = self.place_hubs(cfg, np.random.default_rng(self.hub_seed))
        ids, probs = self._shares()
        n = int(rng.poisson(cfg.lam_per_hour * demand_duration_s / 3600.0))

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
            t_request = float(rng.uniform(0, demand_duration_s))
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


def _uss_rng(base_seed: int, uss_id: str) -> np.random.Generator:
    """Return a stable child RNG for one USS, independent of registry insertion order."""
    digest = hashlib.blake2b(
        uss_id.encode("utf-8"),
        digest_size=8,
        person=b"fspace",
    ).digest()
    uss_key = int.from_bytes(digest, "little")
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(base_seed) & 0xFFFFFFFF,
                (int(base_seed) >> 32) & 0xFFFFFFFF,
                uss_key & 0xFFFFFFFF,
                (uss_key >> 32) & 0xFFFFFFFF,
            ]
        )
    )


def _shift_request_clock(requests: list[FlightRequest], offset_s: float | None = None) -> float:
    """Shift every request and desired departure forward onto a nonnegative clock.

    ``offset_s=None`` (default) shifts by the REALIZED preroll, putting the earliest filing at exactly
    zero. That amount is a max-order statistic over the departure-lead draws, so two otherwise-identical
    runs whose leads differ end up translated relative to each other — every ``t_departure`` moves, and
    the runs can only be compared in aggregate.

    Passing a FIXED ``offset_s`` instead pins the translation, so a family of runs differing only in
    ``departure_offset_s`` keeps byte-identical desired departures and differs solely in FCFS filing
    order — the paired per-flight comparison the scheduling-lead arms rely on. The constant must cover
    the realized preroll: a filing before t=0 would break the planner's monotonic-``t_request``
    occupancy eviction (see ``planner/occupancy.py``), so an undersized offset raises rather than clips.
    """
    if not requests:
        return 0.0
    needed_s = -min(request.t_request for request in requests)
    if offset_s is None:
        shift_s = needed_s
    else:
        shift_s = float(offset_s)
        if shift_s < needed_s:
            raise ValueError(
                f"request_clock_offset_s={shift_s:g} is smaller than the realized preroll "
                f"{needed_s:.1f}s — raise it to at least that (a filing before t=0 breaks the "
                f"monotonic-t_request occupancy eviction)")
    for request in requests:
        request.t_request += shift_s
        request.t_departure += shift_s
    return shift_s


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
      + ``turnaround_s`` in legacy mode. With ``paired_return_request``, both legs are filed together
      and the return's desired departure follows the outbound's nominal arrival. The return's landing
      also consumes a pad, counted against the hub's N.

    ``lam_per_hour`` counts *deliveries*; with returns on, the realised flight count is ~2×. Hubs are
    placed once under ``hub_seed`` (stable infrastructure); only demand varies with ``cfg.seed``.
    """

    n_hubs_per_uss: dict[str, int] = field(
        default_factory=lambda: {"walmart_uss": 6, "stripmall_uss": 20}
    )
    radius_m: "float | dict[str, float]" = 3000.0   # customer demand radius (scalar, or per-USS)
    pads_per_hub: "int | dict[str, int]" = 1         # terminal capacity N per hub (scalar, or per-USS)
    terminal_radius_m: "float | dict[str, float] | None" = None   # column size; None → hover footprint
    corridor_overlap_m: "float | None" = None        # exit-lane overlap into column; None/0 → flush at edge
    return_flights: bool = True                      # each delivery → a return to its origin hub
    turnaround_s: float = 0.0                      # delay before the return is filed (after est. arrival)
    uss_share: dict[str, float] | None = None
    # Per-USS delivery Poisson rate (/hr). When set it REPLACES the global cfg.lam_per_hour × uss_share
    # path entirely: each USS is its own independent Poisson stream (Poisson thinning ⇒ a strict
    # generalization). ``uss_share`` and ``cfg.lam_per_hour`` are then ignored for this model. None ⇒ the
    # legacy global-λ-then-share behaviour (byte-for-byte unchanged).
    lam_per_uss: "dict[str, float] | None" = None
    # Per-USS desired-departure lead as a Gaussian ``(mean_s, std_s)``: a leg filed at t is scheduled to
    # depart at ``t + max(0, N(mean, std))`` (floored at 0 so t_departure ≥ t_request always holds). Set
    # for some USSs to model per-operator scheduling lead / advance booking; absent USSs (or None) depart
    # on filing exactly as today (and draw NO extra randomness). Legacy returns draw their own lead;
    # strategically paired returns inherit their outbound filing and draw no second Gaussian.
    departure_offset_s: "dict[str, tuple[float, float]] | None" = None
    # "request" samples filing times first (legacy behavior); "departure" samples every USS's outbound
    # desired departures over the common demand window, then dynamically pre-rolls filings to time zero.
    timing_mode: str = "request"
    # When true, the outbound and its return are filed together. The return requests departure after the
    # outbound's nominal arrival plus turnaround, with no second scheduling-lead draw.
    paired_return_request: bool = False
    # timing_mode="departure" only: shift the whole clock by this FIXED constant instead of by the
    # realized preroll. Holding it constant across a family of runs that differ only in
    # departure_offset_s gives every arm byte-identical t_departure values, so delays can be compared
    # flight-by-flight rather than only in aggregate. None → the legacy data-dependent shift.
    request_clock_offset_s: "float | None" = None
    min_od_separation_m: float = 200.0
    hub_seed: int = 0xA17F
    min_hub_gap_m: float = 100.0                     # clearance between terminal-airspace EDGES (no overlap)

    def __post_init__(self):
        if self.timing_mode not in {"request", "departure"}:
            raise ValueError(
                f"unknown timing_mode {self.timing_mode!r} (want 'request' | 'departure')")
        if self.request_clock_offset_s is not None:
            # Only the departure-first path shifts the clock at all; silently ignoring the knob in
            # request mode would let an arm family think it was pinned when it never was.
            if self.timing_mode != "departure":
                raise ValueError(
                    "request_clock_offset_s applies only to timing_mode='departure' "
                    f"(got {self.timing_mode!r}); request-first filings are already nonnegative")
            if self.request_clock_offset_s < 0.0:
                raise ValueError(
                    f"request_clock_offset_s must be >= 0 (got {self.request_clock_offset_s})")
        # Fail fast on a mistyped USS key — an experiment silently generating zero flights for a hub is a
        # worse failure mode than a config error at construction.
        hubs = set(self.n_hubs_per_uss)
        for name, d in (("lam_per_uss", self.lam_per_uss),
                        ("departure_offset_s", self.departure_offset_s)):
            unknown = set(d) - hubs if d is not None else set()
            if unknown:
                raise ValueError(
                    f"{name} references USS(es) {sorted(unknown)} absent from "
                    f"n_hubs_per_uss {sorted(hubs)}")

    def place_hubs(self, cfg: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return ``{uss_id: (n_hubs, 2)}`` single-point hub centres, reject-sampled so no two hubs'
        terminal airspaces overlap (:func:`_scatter_hubs`; each USS's column radius is
        ``terminal_radius_m`` or the ``cfg`` hover footprint). DESIGN KNOB for spatial structure (swap
        the uniform scatter for a clustered process to mimic real retail geography)."""
        def radius_of(uid: str) -> float:
            tr = self._terminal_radius_for(uid)
            # always-active terminals wall the WIDER terminal_cells (column + one boundary-hex ring), not
            # just the column — reject-sample on that extent so neighbouring hubs' permanent walls never
            # overlap and foreign-block each other's exit lanes. A boundary-hex centre sits within one hex
            # pitch (SQRT3·circumradius) of the exit_radius edge, so that rigorously upper-bounds the ring
            # for any gap ≥ 0. Flag off ⇒ the bare column radius (transient dwell walls don't engulf).
            if cfg.terminal_airspace_always_active:
                term = Terminal(f"{uid}#0", self._pads_for(uid), tr, self.corridor_overlap_m)
                return exit_radius(term, cfg) + SQRT3 * circumradius(cfg)
            return cfg.terminal_radius_m if tr is None else float(tr)
        return _scatter_hubs(cfg, rng, self.n_hubs_per_uss, radius_of, self.min_hub_gap_m)

    def terminals(self, cfg: SimConfig) -> list:
        """All placed hubs as ``(center, Terminal)`` — permanent vertiport infrastructure, EVERY hub
        regardless of whether it draws a flight this horizon. Under terminal_airspace_always_active the
        sim walls this whole set, matching the foreign-column filter (which drops against ALL placed
        hubs) — else a zero-flight hub would be filtered against but never walled."""
        hubs = self.place_hubs(cfg, np.random.default_rng(self.hub_seed))
        return [(pts[hj], Terminal(f"{uid}#{hj}", self._pads_for(uid),
                                   self._terminal_radius_for(uid), self.corridor_overlap_m))
                for uid, pts in hubs.items() for hj in range(pts.shape[0])]

    def _radius_for(self, uss_id: str) -> float:
        return float(self.radius_m[uss_id] if isinstance(self.radius_m, dict) else self.radius_m)

    def _terminal_radius_for(self, uss_id: str) -> float | None:
        tr = self.terminal_radius_m
        if tr is None:
            return None                              # builder defaults to the hover footprint
        return float(tr[uss_id] if isinstance(tr, dict) else tr)

    def _pads_for(self, uss_id: str) -> int:
        p = self.pads_per_hub
        return int(p[uss_id] if isinstance(p, dict) else p)

    def _shares(self) -> tuple[list[str], np.ndarray]:
        ids = list(self.n_hubs_per_uss)
        p = (np.ones(len(ids)) if self.uss_share is None
             else np.array([self.uss_share.get(uid, 0.0) for uid in ids], float))
        return ids, p / p.sum()

    def _lead_for(self, uss_id: str, rng: np.random.Generator) -> "float | None":
        """Draw a nonnegative scheduling lead, or ``None`` without consuming the RNG."""
        if self.departure_offset_s is None:
            return None
        ms = self.departure_offset_s.get(uss_id)
        if ms is None:
            return None
        mean, std = ms
        return max(0.0, float(rng.normal(mean, std)))

    def _est_trip_s(self, o: np.ndarray, d: np.ndarray, cfg: SimConfig) -> float:
        """Nominal door-to-door time for the return clock: cruise + climb/descent + one pad dwell."""
        dist = float(np.linalg.norm(np.asarray(d, float) - np.asarray(o, float)))
        return dist / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_s + cfg.hover_time_s

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        gl = cfg.ground_level_m
        demand_duration_s = cfg.effective_demand_duration_s
        hubs = self.place_hubs(cfg, np.random.default_rng(self.hub_seed))

        # foreign-column filter (cfg.terminal_airspace_always_active): a delivery whose customer's hex
        # falls inside ANY OTHER hub's permanently-walled terminal cells is unreachable — A* finds the
        # goal hex is_blocked and denies it. Test the EXACT cell set the occupancy walls
        # (hexgrid.terminal_cells, keyed by terminal id just as register_static_terminal builds
        # static_term_cells), so a kept customer's endpoint hex is guaranteed clear for its own flight —
        # no geometric-margin gap. Own hub exempt. terminal_cells resolves radius=None like the wall path.
        filter_foreign = cfg.terminal_airspace_always_active
        if filter_foreign:
            R = circumradius(cfg)
            foreign_cells: dict[tuple[int, int], set] = {}
            for uid, pts in hubs.items():
                for hj in range(pts.shape[0]):
                    term = Terminal(f"{uid}#{hj}", self._pads_for(uid),
                                    self._terminal_radius_for(uid), self.corridor_overlap_m)
                    for cell in terminal_cells(pts[hj], term, cfg):
                        foreign_cells.setdefault(cell, set()).add(term.id)

        requests: list[FlightRequest] = []
        fid = 0

        def emit(uss_id: str, hi: int, event_rng: np.random.Generator) -> None:
            """Emit one outbound delivery and, optionally, its return."""
            nonlocal fid
            hub = hubs[uss_id][hi]
            terminal = Terminal(f"{uss_id}#{hi}", self._pads_for(uss_id),
                                self._terminal_radius_for(uss_id), self.corridor_overlap_m)
            radius = self._radius_for(uss_id)
            # Keep customers clear of the hub's own always-active WALL. A customer within
            # (terminal_radius + hover_radius) of its serving hub has its landing/takeoff column overlap
            # the wall — and that column is UNTAGGED (a customer, not the hub), so it is not exempt from the
            # wall (conflict.volumes_conflict) → a spurious conflict_filed that denies the delivery+return.
            # Require the customer ≥ 1.5× the hub's terminal (wall) radius away (and never below the existing
            # min_od_separation_m floor), which clears the wall+column overlap for every hub size. Assumes the
            # service ``radius`` exceeds ``min_r`` (true for every configured scenario: radius ≫ terminal_radius);
            # otherwise the redraw loop can't satisfy it and falls back to a clipped, possibly too-close point.
            min_r = max(self.min_od_separation_m, 1.5 * terminal_radius(terminal, cfg))
            customer = None
            for _ in range(20):  # redraw until in-region and clear of the hub's wall footprint
                c = _sample_in_disk(hub, radius, event_rng)
                if 0.0 <= c[0] <= w and 0.0 <= c[1] <= h and \
                        np.linalg.norm(c - hub) >= min_r:
                    customer = c
                    break
            if customer is None:
                customer = np.clip(c, [0.0, 0.0], [w, h])
            sampled_clock = float(event_rng.uniform(0, demand_duration_s))
            lead_s = self._lead_for(uss_id, event_rng)
            if self.timing_mode == "departure":
                t_dep = sampled_clock
                t_req = t_dep - (0.0 if lead_s is None else lead_s)
            else:
                t_req = sampled_clock
                t_dep = t_req + (0.0 if lead_s is None else lead_s)
            drop = False
            if filter_foreign:                                       # customer hex inside a FOREIGN wall?
                walls = foreign_cells.get(enu_to_axial(customer[0], customer[1], R))
                # own column is transparent — drop (both legs) only if a FOREIGN terminal walls the hex
                drop = walls is not None and any(tid != terminal.id for tid in walls)
            if not drop:
                requests.append(FlightRequest(                        # delivery: hub → customer
                    fid, vec(hub[0], hub[1], gl), vec(customer[0], customer[1], gl), t_req,
                    t_departure=t_dep, uss_id=uss_id, origin_terminal=terminal))
            outbound_fid = fid
            fid += 1
            if self.return_flights:                                  # return: customer → same hub
                trip_and_turnaround_s = (
                    self._est_trip_s(hub, customer, cfg) + self.turnaround_s
                )
                if self.paired_return_request:
                    t_ret = t_req
                    t_ret_dep = t_dep + trip_and_turnaround_s
                else:
                    # Legacy request-first semantics: file the return after the nominal outbound arrival,
                    # then give it its own independently sampled scheduling lead.
                    t_ret = t_req + trip_and_turnaround_s
                    return_lead_s = self._lead_for(uss_id, event_rng)
                    t_ret_dep = t_ret + (0.0 if return_lead_s is None else return_lead_s)
                if not drop:                                          # foreign-column filter drops both legs
                    requests.append(FlightRequest(
                        fid, vec(customer[0], customer[1], gl), vec(hub[0], hub[1], gl), t_ret,
                        t_departure=t_ret_dep, uss_id=uss_id, dest_terminal=terminal,
                        paired_outbound_id=outbound_fid))
                fid += 1

        if self.lam_per_uss is None:
            # Legacy path: one global Poisson count, each flight's USS drawn from uss_share.
            ids, probs = self._shares()
            n = int(rng.poisson(cfg.lam_per_hour * demand_duration_s / 3600.0))
            for _ in range(n):
                uss_id = ids[int(rng.choice(len(ids), p=probs))]
                emit(uss_id, int(rng.integers(hubs[uss_id].shape[0])), rng)
        else:
            # Per-USS path: an independent Poisson stream per operator; cfg.lam_per_hour / uss_share unused.
            # Departure-first scenarios receive stable child streams, so adding another USS cannot perturb
            # an existing operator's candidate sequence. Request-first mode retains the legacy shared stream.
            base_seed = (
                int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
                if self.timing_mode == "departure"
                else None
            )
            for uss_id in self.n_hubs_per_uss:
                # Omitting a hub USS from lam_per_uss is ALLOWED and means zero demand (infrastructure-only
                # hub). A MISTYPED key is not: __post_init__ rejects lam_per_uss keys absent from
                # n_hubs_per_uss, so the only way to reach this default-0 is a deliberate omission.
                lam = float(self.lam_per_uss.get(uss_id, 0.0))
                uss_rng = _uss_rng(base_seed, uss_id) if base_seed is not None else rng
                n_uss = int(uss_rng.poisson(lam * demand_duration_s / 3600.0))
                for _ in range(n_uss):
                    emit(
                        uss_id,
                        int(uss_rng.integers(hubs[uss_id].shape[0])),
                        uss_rng,
                    )

        if self.timing_mode == "departure":
            _shift_request_clock(requests, self.request_clock_offset_s)
        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests
