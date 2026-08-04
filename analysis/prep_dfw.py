"""Bake the DFW geo artifacts the `dfw_*` scenarios read at runtime.

ONE-TIME offline step (needs the `geo` extra: geopandas/pyogrio/openpyxl/pyproj/shapely). It ingests
the three source files the user provides in `.context/dfw_source/` and writes three small, committed
artifacts under `freespace_sim/data/dfw/` — in lon/lat, so re-framing the region needs no re-bake:

  retail_pois.csv       lon, lat, category, pop_density     (Overture POIs; the wing/zipline hub pool)
  amazon_facilities.csv lon, lat, code, type, pop_density   (fixed Amazon hubs)
  tracts.npz            population + ragged polygon rings    (census-density destination sampling)

`pop_density` is the containing Census tract's ACS 2018-2022 population density (people / sq-mi),
computed exactly as the user's reference choropleth script (area via EPSG:32139). Mirrors that script.

Run (direct path, matching the repo's analysis/ convention):
  uv run --extra geo python analysis/prep_dfw.py \
      --gdb .context/dfw_source/ACS_2022_5YR_TRACT_48_TEXAS.gdb.zip \
      --overture .context/dfw_source/overture_retail.csv \
      --amazon .context/dfw_source/amazon_dfw_facilities.xlsx \
      --out freespace_sim/data/dfw
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- Census ACS geodatabase layers (2022 5-year, Texas) — same as the reference script ---
GEO_LAYER = "ACS_2022_5YR_TRACT_48_TEXAS"          # tract polygons (COUNTYFP, GEOIDFQ)
POP_LAYER, POP_FIELD = "X01_AGE_AND_SEX", "B01001_E001"   # total population, joined on GEOIDFQ
SQMI_PER_SQM = 1.0 / 2_589_988.110336              # m² → sq-mi (density = people / sq-mi)
AREA_EPSG = 32139                                  # NAD83 / Texas North Central (metres) for area

# Wide DFW frame (reference display window) — clip tracts here so any region within the metro is
# covered without a re-bake. Overture/Amazon inputs are already inside it.
MINLON, MAXLON, MINLAT, MAXLAT = -97.9767, -95.9240, 32.1788, 33.5030

# Region window the dfw_* scenarios actually run (SimConfig default centre + density REGION_M), used
# only for the printed feasibility read-out. Keep in sync with config.region_center_latlon / density.REGION_M.
REGION_CENTER_LATLON = (32.90, -97.04)
REGION_M = (60_000.0, 30_000.0)
# Retail categories the wing/zipline hubs are sampled from (keep in sync with scenarios/dfw.HUB_CATEGORIES).
HUB_CATEGORIES = ("shopping_center", "shopping", "mall", "department_store", "grocery_store",
                  "discount_store", "home_improvement_store", "building_supply_store")


def _gdb_dataset(path: str) -> str:
    """Return a pyogrio-openable dataset path, reading a zipped `.gdb` in place via GDAL `/vsizip/`."""
    if not path.endswith(".zip"):
        return path
    with zipfile.ZipFile(path) as z:
        gdbs = sorted({n.split("/", 1)[0] for n in z.namelist() if ".gdb" in n})
    inner = next((d for d in gdbs if d.endswith(".gdb")), None)
    if inner is None:
        raise SystemExit(f"no *.gdb directory found inside {path}")
    return f"/vsizip/{os.path.abspath(path)}/{inner}"


def _region_bbox_latlon() -> tuple[float, float, float, float]:
    """Region lon/lat bbox from centre + size (equirectangular), for the feasibility read-out."""
    lat0, lon0 = REGION_CENTER_LATLON
    dlat = (REGION_M[1] / 2.0) / 111_320.0
    dlon = (REGION_M[0] / 2.0) / (111_320.0 * np.cos(np.radians(lat0)))
    return lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat


def _tracts_to_ragged(gdf) -> dict:
    """Flatten (multi)polygon tract geometries into offset-indexed lon/lat rings (no pickle)."""
    poly_chunks: list[np.ndarray] = []
    ring_offsets = [0]
    tract_ring_offsets = [0]
    population: list[float] = []
    for geom, pop in zip(gdf.geometry.values, gdf["population"].values):
        if geom is None or geom.is_empty:
            continue
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        nrings = 0
        for poly in polys:
            for ring in (poly.exterior, *poly.interiors):
                coords = np.asarray(ring.coords, dtype=float)[:, :2]   # (Mi, 2) lon, lat
                if len(coords) < 3:
                    continue
                poly_chunks.append(coords)
                ring_offsets.append(ring_offsets[-1] + len(coords))
                nrings += 1
        if nrings == 0:
            continue
        tract_ring_offsets.append(tract_ring_offsets[-1] + nrings)
        population.append(float(pop))
    return {
        "population": np.asarray(population, dtype=float),
        "poly_lonlat": np.vstack(poly_chunks),
        "ring_offsets": np.asarray(ring_offsets, dtype=np.int64),
        "tract_ring_offsets": np.asarray(tract_ring_offsets, dtype=np.int64),
    }


def main(gdb: str, overture_csv: str, amazon_xlsx: str, out_dir: str, simplify_deg: float = 3e-4) -> None:
    import geopandas as gpd
    import pyogrio

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = _gdb_dataset(gdb)

    # --- tracts + population density (mirrors the reference choropleth script) ---
    gdf = pyogrio.read_dataframe(ds, layer=GEO_LAYER, columns=["COUNTYFP", "GEOIDFQ"])
    gdf = gdf.to_crs(4326)                                        # normalise to lon/lat
    gdf = gdf.cx[MINLON:MAXLON, MINLAT:MAXLAT].copy()             # clip to the wide DFW frame
    pop = pyogrio.read_dataframe(ds, layer=POP_LAYER, read_geometry=False,
                                 columns=["GEOIDFQ", POP_FIELD]).rename(columns={POP_FIELD: "population"})
    gdf = gdf.merge(pop, on="GEOIDFQ", how="left")
    gdf["population"] = gdf["population"].fillna(0).astype(float)
    gdf = gdf[gdf["population"] > 0].copy()
    gdf["density"] = gdf["population"] / (gdf.to_crs(AREA_EPSG).geometry.area * SQMI_PER_SQM)
    gdf["geometry"] = gdf.geometry.simplify(simplify_deg, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    tracts = gpd.GeoDataFrame(gdf[["density", "population", "geometry"]], crs=4326)

    def _join_density(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.Series:
        pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326)
        joined = gpd.sjoin(pts, tracts[["density", "geometry"]], how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]    # a point tiles into ≤1 tract
        return joined["density"].fillna(0.0).to_numpy()

    # --- Overture retail POIs (the wing/zipline hub pool) ---
    poi = pd.read_csv(overture_csv)
    poi["pop_density"] = _join_density(poi, "lon", "lat")
    poi[["lon", "lat", "category", "pop_density"]].to_csv(out / "retail_pois.csv", index=False)

    # --- Amazon facilities (fixed hubs) — record ALL types ---
    fac = pd.ExcelFile(amazon_xlsx).parse("Facilities").dropna(subset=["Latitude", "Longitude"])
    fac["pop_density"] = _join_density(fac, "Longitude", "Latitude")
    fac_out = fac.rename(columns={"Longitude": "lon", "Latitude": "lat", "Code": "code", "Type": "type"})
    fac_out[["lon", "lat", "code", "type", "pop_density"]].to_csv(out / "amazon_facilities.csv", index=False)

    # --- tracts.npz (ragged, offset-indexed rings) ---
    ragged = _tracts_to_ragged(tracts)
    np.savez_compressed(out / "tracts.npz", **ragged)

    # --- feasibility read-out over the actual 60×30 km region window ---
    lo_lon, hi_lon, lo_lat, hi_lat = _region_bbox_latlon()
    in_box = poi["lon"].between(lo_lon, hi_lon) & poi["lat"].between(lo_lat, hi_lat)
    cand = poi[in_box & poi["category"].isin(HUB_CATEGORIES)]
    print(f"tracts baked: {len(ragged['population'])} (pop>0, wide frame) | "
          f"vertices: {len(ragged['poly_lonlat']):,}")
    print(f"retail POIs: {len(poi):,} | amazon facilities: {len(fac_out)} "
          f"({fac_out['type'].value_counts().to_dict()})")
    print(f"hub candidates in the 60x30 km region ({len(cand)} total, need up to 476):")
    for c, n in cand["category"].value_counts().items():
        print(f"    {n:5d}  {c}")
    print(f"wrote: {out/'retail_pois.csv'}, {out/'amazon_facilities.csv'}, {out/'tracts.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bake DFW geo artifacts for the dfw_* scenarios.")
    ap.add_argument("--gdb", required=True, help="ACS tract geodatabase (.gdb dir or .gdb.zip)")
    ap.add_argument("--overture", required=True, help="Overture retail CSV (id,name,category,lon,lat,...)")
    ap.add_argument("--amazon", required=True, help="Amazon facilities .xlsx (Facilities sheet)")
    ap.add_argument("--out", default="freespace_sim/data/dfw", help="artifact output directory")
    ap.add_argument("--simplify", type=float, default=3e-4, help="tract geometry simplify tolerance (deg)")
    a = ap.parse_args()
    main(a.gdb, a.overture, a.amazon, a.out, simplify_deg=a.simplify)
