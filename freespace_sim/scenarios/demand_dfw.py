"""DFW real-geography demand — hub locations and destinations grounded in Overture + Census data.

:class:`DfwGeoDemand` is a thin subclass of :class:`~freespace_sim.demand.HubRadiusDemand` that swaps
ONLY the *spatial* generation, keeping every other knob (per-USS Poisson streams, paired returns,
departure-lead timing, multi-pad terminals, always-active walls, foreign-column filter) identical — so
a ``dfw_*`` scenario is its ``density_*`` twin with real geography:

- **place_hubs** — *sampled* operators (wing/zipline) draw their hub sites from real Overture retail
  POIs weighted by Census tract population density; *fixed* operators (Amazon) sit at real facility
  coordinates. Both respect the same non-overlap separation contract as the synthetic scatter
  (:func:`freespace_sim.demand._scatter_hubs`), fixed anchors placed first.
- **_draw_customer** — a delivery's customer is drawn by sampling a Census tract ∝ population, then a
  uniform point inside it (clipped to the hub's service disk), so destinations follow population density.

All geodata is read (pure numpy) and projected once via :func:`freespace_sim.geo.load_dfw_geo`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..demand import HubRadiusDemand
from ..geo import load_dfw_geo, point_in_polygon

# Retail categories eligible as wing/zipline hub sites. The FAA EAs approve "shopping centers, large
# individual retailers, and shopping malls" — the large-format retailers (department/grocery/discount/
# home-improvement) both fit that language and supply enough candidates for the 476-hub future world.
DEFAULT_HUB_CATEGORIES = ("shopping_center", "shopping", "mall", "department_store", "grocery_store",
                          "discount_store", "home_improvement_store", "building_supply_store")
DEFAULT_FIXED_TYPES = ("Delivery station", "Same-day site")   # Amazon last-mile facilities


@dataclass
class DfwGeoDemand(HubRadiusDemand):
    dataset: str = "dfw"                                        # artifact dir under freespace_sim/data/
    sampled_hub_uss: tuple = ("wing_zipline_uss",)             # hubs ∝ tract density from retail POIs
    fixed_hub_uss: tuple = ("amazon_uss",)                     # hubs at real facility coordinates
    fixed_hub_types: tuple = DEFAULT_FIXED_TYPES               # which facility types are drone hubs
    hub_categories: tuple = DEFAULT_HUB_CATEGORIES             # retail categories eligible as hub sites
    use_all_fixed_hubs: bool = False                          # True → every listed facility (ignore count)
    geo_max_attempts: int = 40                                # tract draws before the disk fallback
    geo_point_attempts: int = 20                              # in-tract point draws per tract draw

    def __post_init__(self):
        super().__post_init__()
        hubs = set(self.n_hubs_per_uss)
        unknown = (set(self.sampled_hub_uss) | set(self.fixed_hub_uss)) - hubs
        if unknown:
            raise ValueError(f"sampled/fixed_hub_uss {sorted(unknown)} absent from "
                             f"n_hubs_per_uss {sorted(hubs)}")
        if set(self.sampled_hub_uss) & set(self.fixed_hub_uss):
            raise ValueError("an operator cannot be both sampled and fixed")
        orphan = hubs - set(self.sampled_hub_uss) - set(self.fixed_hub_uss)
        if orphan:
            raise ValueError(f"USS {sorted(orphan)} has no hub source (neither sampled nor fixed)")

    # ---- geodata (loaded + projected once, cached on the instance) ----
    def _geo(self, cfg):
        key = (self.dataset, tuple(map(float, cfg.region_center_latlon)),
               tuple(map(float, cfg.region_size_m)))
        cached = getattr(self, "_geo_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, load_dfw_geo(self.dataset, cfg))
            self._geo_cache = cached
        return cached[1]

    def _sep_radius(self, uid: str, cfg) -> float:
        """The non-overlap radius for ``uid``'s hubs — the shared base wall radius."""
        return self._wall_radius(uid, cfg)

    def _separated(self, c, r: float, placed: list) -> bool:
        """True iff centre ``c`` (radius ``r``) clears every placed hub by ``r_i+r_j+min_hub_gap_m``."""
        for cj, rj in placed:
            need = r + rj + self.min_hub_gap_m
            if (c[0] - cj[0]) ** 2 + (c[1] - cj[1]) ** 2 < need * need:
                return False
        return True

    def place_hubs(self, cfg, rng):
        """{uss_id: (n, 2)} hub centres in ENU metres — fixed Amazon anchors first, then density-
        weighted wing/zipline reject-sampled against the shared placed set (stable in ``hub_seed``)."""
        geo = self._geo(cfg)
        placed: list = []                                      # (centre, radius) across ALL operators
        out: dict = {}

        for uid in self.fixed_hub_uss:                         # real facilities, densest tract first
            r = self._sep_radius(uid, cfg)
            xy, w = geo.amazon_of_types(self.fixed_hub_types)
            order = np.argsort(-w, kind="stable")
            cap = len(order) if self.use_all_fixed_hubs else int(self.n_hubs_per_uss[uid])
            picks = []
            for i in order:
                if len(picks) >= cap:
                    break
                if self._separated(xy[i], r, placed):
                    picks.append(int(i)); placed.append((xy[i], r))
            out[uid] = xy[picks] if picks else np.empty((0, 2), float)
            if len(out[uid]) == 0 and float((self.lam_per_uss or {}).get(uid, 0.0)) > 0.0:
                raise ValueError(f"no fixed hubs of types {self.fixed_hub_types} for {uid!r} survived "
                                 f"placement, but it has demand — widen fixed_hub_types or the region")

        for uid in self.sampled_hub_uss:                       # retail POIs ∝ pop_density, no replacement
            r = self._sep_radius(uid, cfg)
            xy, w = geo.pois_of_categories(self.hub_categories)
            n_target = int(self.n_hubs_per_uss[uid])
            avail = np.ones(len(xy), dtype=bool)
            picks = []
            while len(picks) < n_target:
                if not avail.any() or w[avail].sum() <= 0.0:
                    raise ValueError(f"retail pool exhausted for {uid!r}: {len(picks)}/{n_target} hubs "
                                     f"placed (widen hub_categories or enlarge the region)")
                idx = np.flatnonzero(avail)
                i = int(rng.choice(idx, p=w[idx] / w[idx].sum()))
                avail[i] = False
                if self._separated(xy[i], r, placed):
                    picks.append(i); placed.append((xy[i], r))
            out[uid] = xy[picks]
        return out

    # ---- destinations: census-tract population density ----
    def _hub_candidates(self, hub, radius: float, cfg):
        """Cached ``(tract_indices, population_probabilities)`` for populated tracts whose bbox meets
        the hub's service disk. The normalised ``p`` is cached with the index set — it is redrawn for
        every delivery but there are only ever as many distinct (hub, radius) keys as there are hubs."""
        cache = getattr(self, "_cand_cache", None)
        if cache is None:
            cache = self._cand_cache = {}
        key = (round(float(hub[0]), 2), round(float(hub[1]), 2), round(float(radius), 2))
        hit = cache.get(key)
        if hit is None:
            geo = self._geo(cfg)
            hx, hy = float(hub[0]), float(hub[1])
            bb = geo.tract_bbox
            m = ((bb[:, 0] <= hx + radius) & (bb[:, 2] >= hx - radius)
                 & (bb[:, 1] <= hy + radius) & (bb[:, 3] >= hy - radius)
                 & (geo.tract_pop > 0.0))
            idx = np.flatnonzero(m)
            pop = geo.tract_pop[idx]
            hit = (idx, pop / pop.sum() if len(idx) else pop)
            cache[key] = hit
        return hit

    def _draw_customer(self, hub, radius, min_r, w, h, cfg, event_rng):
        """Sample a tract ∝ population, then a uniform point inside THAT tract, restarting the whole
        draw if the point misses the hub's service annulus or the region — so a tract's chance of
        being used is ``pop × (its area inside the annulus / its total area)``, i.e. destinations
        follow population density exactly. Falls back to the base uniform-in-disk draw if the sampler
        can't seat a point at all (never fails to emit; zero fallbacks over a full dfw_future run).

        The two loops are NOT interchangeable. Redrawing the *tract* on an in-tract rejection (rather
        than only the point) re-weights every tract by how much of its bounding box the polygon fills
        — 0.18-0.99 across the baked DFW tracts, uncorrelated with population — which sampled compact
        tracts up to 5.7x more per resident than sprawling ones. Only rejections that genuinely carve
        area off a tract (disk, ``min_r``, region) may restart the tract draw."""
        geo = self._geo(cfg)
        idx, p = self._hub_candidates(hub, radius, cfg)
        if len(idx):
            hx, hy = float(hub[0]), float(hub[1])
            r2, min_r2 = radius * radius, min_r * min_r
            for _ in range(self.geo_max_attempts):
                t = int(idx[event_rng.choice(len(idx), p=p)])         # tract ∝ population
                xmin, ymin, xmax, ymax = geo.tract_bbox[t]
                for _ in range(self.geo_point_attempts):              # uniform INSIDE the drawn tract
                    cx = float(event_rng.uniform(xmin, xmax))
                    cy = float(event_rng.uniform(ymin, ymax))
                    if point_in_polygon(cx, cy, geo.tract_rings[t]):
                        break
                else:
                    continue                                          # sliver tract — draw another
                d2 = (cx - hx) ** 2 + (cy - hy) ** 2
                if min_r2 <= d2 <= r2 and 0.0 <= cx <= w and 0.0 <= cy <= h:
                    return np.array([cx, cy], dtype=float)
        return super()._draw_customer(hub, radius, min_r, w, h, cfg, event_rng)
