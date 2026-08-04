"""Real-geography ingestion for the DFW scenarios — pure numpy, no geo libraries at runtime.

The heavy geodata work (Census tract density, Overture POI join, projection to an equal-area CRS) is
done ONCE, offline, by ``analysis/prep_dfw.py`` (which needs the ``geo`` extra: geopandas/pyogrio/…).
It bakes three small, committed artifacts under ``freespace_sim/data/<dataset>/`` in **lon/lat**:

- ``retail_pois.csv``     — ``lon, lat, category, pop_density`` (every Overture retail POI; the hub pool)
- ``amazon_facilities.csv`` — ``lon, lat, code, type, pop_density`` (fixed Amazon hubs)
- ``tracts.npz``          — Census tracts: ``population`` + ragged polygon rings (no pickle)

This module reads those with numpy/pandas only and PROJECTS them into the simulator's local ENU-metre
frame at load time — a dependency-free equirectangular map anchored at ``cfg.region_center_latlon``
(finally making that field load-bearing). :class:`~freespace_sim.scenarios.demand_dfw.DfwGeoDemand` consumes the
result. Nothing here imports shapely/geopandas, so plain ``uv run`` never pulls GDAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_EARTH_R_M = 6_371_000.0
_DATA_ROOT = Path(__file__).resolve().parent / "data"


def project_lonlat_to_enu(
    lon: np.ndarray, lat: np.ndarray, lat0: float, lon0: float, w: float, h: float
) -> np.ndarray:
    """Equirectangular lon/lat → local ENU metres, mapping the anchor ``(lat0, lon0)`` to the region
    CENTRE ``(w/2, h/2)`` (+east/+north). Distortion is < 0.3 % over a ~60 km metro window, and the
    map is monotone per axis (x depends only on lon, y only on lat), so a projected bbox stays a bbox.

    Accepts scalars or arrays; returns ``(..., 2)`` stacked ``[x, y]``.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * _EARTH_R_M + w / 2.0
    y = np.radians(lat - lat0) * _EARTH_R_M + h / 2.0
    return np.stack([x, y], axis=-1)


def point_in_polygon(px: float, py: float, rings: list[np.ndarray]) -> bool:
    """Even-odd ray-cast point-in-polygon over ALL ``rings`` of a (multi)polygon.

    ``rings`` is a list of ``(M, 2)`` vertex arrays — exterior(s) and holes together. Counting
    crossings of a +x ray across every ring and taking the parity handles holes and disjoint
    multipolygon pieces without any per-ring hole/exterior bookkeeping (a point in a hole crosses two
    boundaries → even → outside). Robust to closed or open rings (a closing zero-length edge never
    crosses). Vectorised over each ring's edges; scalar in the query point.
    """
    crossings = 0
    for ring in rings:
        x = ring[:, 0]
        y = ring[:, 1]
        x1 = np.roll(x, -1)
        y1 = np.roll(y, -1)
        straddles = (y > py) != (y1 > py)                       # edge crosses the horizontal ray line
        denom = np.where(straddles, y1 - y, 1.0)                # safe: masked out where not straddling
        x_int = (x1 - x) * (py - y) / denom + x                 # x of intersection with the ray line
        crossings += int(np.count_nonzero(straddles & (px < x_int)))
    return (crossings % 2) == 1


def _in_region(xy: np.ndarray, w: float, h: float) -> np.ndarray:
    """Boolean mask: point strictly within the ``[0, w] × [0, h]`` region box (inclusive edges)."""
    return (xy[:, 0] >= 0.0) & (xy[:, 0] <= w) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= h)


@dataclass(frozen=True)
class DfwGeo:
    """DFW geodata projected into one region's ENU frame (out-of-region points already DROPPED).

    ``pois_*``/``amazon_*`` are the hub candidate pools; ``tract_*`` drives density-weighted destination
    sampling (``tract_rings[t]`` is a list of ENU ring arrays; ``tract_bbox`` is ``[xmin,ymin,xmax,ymax]``).
    """

    pois_xy: np.ndarray          # (M, 2) ENU
    pois_cat: np.ndarray         # (M,) category strings
    pois_w: np.ndarray           # (M,) tract population density (hub sampling weight)
    amazon_xy: np.ndarray        # (K, 2) ENU
    amazon_type: np.ndarray      # (K,) facility type strings
    amazon_w: np.ndarray         # (K,) tract population density
    tract_pop: np.ndarray        # (T,) population
    tract_bbox: np.ndarray       # (T, 4) ENU [xmin, ymin, xmax, ymax]
    tract_rings: list            # list[T] of list[(Mi, 2)] ENU ring arrays

    def pois_of_categories(self, categories) -> tuple[np.ndarray, np.ndarray]:
        """``(xy, weight)`` for POIs whose category is in ``categories`` (the wing/zipline hub pool)."""
        keep = np.isin(self.pois_cat, list(categories))
        return self.pois_xy[keep], self.pois_w[keep]

    def amazon_of_types(self, types) -> tuple[np.ndarray, np.ndarray]:
        """``(xy, weight)`` for Amazon facilities whose type is in ``types`` (fixed hubs)."""
        keep = np.isin(self.amazon_type, list(types))
        return self.amazon_xy[keep], self.amazon_w[keep]


def load_dfw_geo(dataset: str, cfg) -> DfwGeo:
    """Read the baked ``<dataset>`` artifacts and project them into ``cfg``'s ENU frame.

    Out-of-region POIs and facilities are DROPPED (never clipped — a real site is never silently
    relocated). Raises ``FileNotFoundError`` with a hint to run ``analysis/prep_dfw.py`` if the
    artifacts are missing.
    """
    base = _DATA_ROOT / dataset
    if not base.is_dir():
        raise FileNotFoundError(
            f"DFW geo artifacts not found at {base} — run "
            f"`uv run --extra geo python analysis/prep_dfw.py --out freespace_sim/data/{dataset} ...` first")
    w, h = float(cfg.region_size_m[0]), float(cfg.region_size_m[1])
    lat0, lon0 = float(cfg.region_center_latlon[0]), float(cfg.region_center_latlon[1])

    poi = pd.read_csv(base / "retail_pois.csv")
    pxy = project_lonlat_to_enu(poi["lon"].to_numpy(), poi["lat"].to_numpy(), lat0, lon0, w, h)
    pin = _in_region(pxy, w, h)

    fac = pd.read_csv(base / "amazon_facilities.csv")
    fxy = project_lonlat_to_enu(fac["lon"].to_numpy(), fac["lat"].to_numpy(), lat0, lon0, w, h)
    fin = _in_region(fxy, w, h)

    npz = np.load(base / "tracts.npz")
    tract_pop = npz["population"].astype(float)
    poly_xy = project_lonlat_to_enu(
        npz["poly_lonlat"][:, 0], npz["poly_lonlat"][:, 1], lat0, lon0, w, h)  # (P, 2) ENU
    ring_off = npz["ring_offsets"]           # (R+1,) vertex offsets per ring
    tract_ring_off = npz["tract_ring_offsets"]  # (T+1,) ring offsets per tract
    tract_rings: list = []
    tract_bbox = np.empty((len(tract_pop), 4), dtype=float)
    for t in range(len(tract_pop)):
        rings = [poly_xy[ring_off[r]:ring_off[r + 1]]
                 for r in range(tract_ring_off[t], tract_ring_off[t + 1])]
        tract_rings.append(rings)
        pts = np.vstack(rings)
        tract_bbox[t] = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    return DfwGeo(
        pois_xy=pxy[pin], pois_cat=poi["category"].to_numpy()[pin], pois_w=poi["pop_density"].to_numpy()[pin],
        amazon_xy=fxy[fin], amazon_type=fac["type"].to_numpy()[fin], amazon_w=fac["pop_density"].to_numpy()[fin],
        tract_pop=tract_pop, tract_bbox=tract_bbox, tract_rings=tract_rings,
    )
