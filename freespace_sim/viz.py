"""Visualization — top-down snapshots, a 3D trimesh scene, and a congestion heatmap.

Everything renders the *exact* committed geometry (`BoxSpec`/`CylinderSpec`), so the picture is the
ledger, not an approximation of it. Colors step by the golden ratio (as in the sibling project) so
adjacent flight ids stay visually distinct. matplotlib runs headless (Agg) — figures go to files.
"""

from __future__ import annotations

import colorsys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from .geometry import BoxSpec, CylinderSpec  # noqa: E402
from .sim import SimResult  # noqa: E402
from .volumes import Volume4D  # noqa: E402

_GOLDEN = 0.618033988749895


def flight_color(flight_id: int) -> tuple[float, float, float]:
    """Visually-distinct RGB for a flight, hue stepped by the golden ratio (legacy / single-USS)."""
    h = (flight_id * _GOLDEN) % 1.0
    return colorsys.hsv_to_rgb(h, 0.62, 0.95)


def uss_hues(uss_ids) -> dict[str, float]:
    """Assign each USS an evenly-spaced base hue (deterministic by sorted id). With two operators
    these land on opposite sides of the wheel → maximally distinct families."""
    ids = sorted(set(uss_ids))
    n = max(1, len(ids))
    return {uid: i / n for i, uid in enumerate(ids)}


def flight_color_by_uss(uss_id: str, flight_id: int, hues: dict[str, float]) -> tuple[float, float, float]:
    """RGB for a flight: hue identifies the owning USS, while saturation/value jitter by ``flight_id``
    so same-operator flights read as a distinguishable family rather than one flat blob."""
    h = hues.get(uss_id, 0.0)
    s = 0.45 + 0.30 * ((flight_id * _GOLDEN) % 1.0)
    v = 0.78 + 0.20 * ((flight_id * 0.387) % 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def result_uss_hues(result: SimResult) -> dict[str, float]:
    """Stable hue map over every USS present in the run (accepted *or* denied) so colors don't shift
    between snapshots when a given operator has no active flight at some ``t``."""
    return uss_hues({i.request.uss_id for i in result.intents})


def uss_swatch_hex(uss_id: str, hues: dict[str, float]) -> str:
    """Canonical legend swatch (the USS's base hue at fixed sat/value) as a #rrggbb string."""
    r, g, b = colorsys.hsv_to_rgb(hues.get(uss_id, 0.0), 0.6, 0.9)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def box_footprint(spec: BoxSpec) -> np.ndarray:
    """The 4 xy corners (CCW) of an oriented corridor box, projected to the ground plane."""
    rot = spec.rotation()
    c = np.array(spec.center, float)
    lx, wy, _ = spec.extents
    local = np.array([[lx / 2, wy / 2, 0], [lx / 2, -wy / 2, 0],
                      [-lx / 2, -wy / 2, 0], [-lx / 2, wy / 2, 0]], float)
    return ((rot @ local.T).T + c)[:, :2]


def _active(volumes: list[Volume4D] | None, t: float) -> list[Volume4D]:
    """Volumes whose half-open window ``[t_start, t_end)`` is live at time ``t``."""
    return [v for v in (volumes or []) if v.t_start <= t < v.t_end]


def _position_at(centerline, t: float):
    """Linear-interpolate the drone (x, y) on its centerline at time t, or None if not airborne."""
    if not centerline or t < centerline[0][1] or t > centerline[-1][1]:
        return None
    for (p0, t0), (p1, t1) in zip(centerline, centerline[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
            return (p0 + f * (p1 - p0))[:2]
    return centerline[-1][0][:2]


def snapshot(result: SimResult, t: float, ax=None, out=None, uss: str | None = None):
    """Top-down view of every reservation active at time ``t`` + drone dots, colored by owning USS.

    Renders the committed geometry exactly: corridor boxes as translucent polygons, hover cylinders
    as circles, each airborne drone as a dot on its centerline.

    Parameters
    ------------
    - result (SimResult): run whose accepted intents are drawn.
    - t (float): simulation time (s) to snapshot.
    - ax (Axes | None): target matplotlib axes; a new 8x8 figure is made when None.
    - out (path | None): if set, the figure is saved here (120 dpi) and closed.
    - uss (str | None): restrict to one operator's flights; None draws all.

    Return
    --------
    - output (Axes): the axes the snapshot was drawn on.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8, 8))
    hues = result_uss_hues(result)
    shown = [i for i in result.accepted if uss is None or i.request.uss_id == uss]
    for intent in shown:
        col = flight_color_by_uss(intent.request.uss_id, intent.request.flight_id, hues)
        for v in _active(intent.volumes, t):
            if isinstance(v.shape, BoxSpec):
                ax.add_patch(Polygon(box_footprint(v.shape), closed=True,
                                     facecolor=col, edgecolor=col, alpha=0.35, lw=0.5))
            elif isinstance(v.shape, CylinderSpec):
                ax.add_patch(Circle((v.shape.cx, v.shape.cy), v.shape.radius,
                                    facecolor=col, edgecolor=col, alpha=0.20, lw=0.8))
        pos = _position_at(intent.centerline, t)
        if pos is not None:
            ax.plot(pos[0], pos[1], "o", color=col, ms=5, mec="k", mew=0.4)

    w, h = result.config.region_size_m
    ax.set_xlim(-0.05 * w, w)
    ax.set_ylim(-0.05 * h, h)
    ax.set_aspect("equal")
    scope = "" if uss is None else f"   ·   USS={uss}"
    ax.set_title(f"t = {t:.0f} s{scope}   ·   active flights: "
                 f"{sum(bool(_active(i.volumes, t)) for i in shown)}")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    if out:
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def congestion_heatmap(result: SimResult, out=None, bins: int = 60):
    """2D histogram of reserved volume-seconds projected onto the ground plane (where airspace is
    busiest) — the free-space analog of the sibling's hex-occupancy heatmap.

    Integrated over :func:`metrics.simulation_window` (the same span ``metrics.flight_frame`` uses),
    so the figure's total reconciles with ``summary.json``'s ``reserved_vol_m3_s``. Do NOT anchor on
    ``cfg.horizon_s`` instead: it drops the post-horizon return tail the density scenarios produce.

    Parameters
    ------------
    - result (SimResult): run whose accepted intents are summed.
    - out (path | None): if set, the figure is saved here (120 dpi) and closed.
    - bins (int): grid resolution per axis of the ground-plane histogram.

    Return
    --------
    - output (Axes): the axes holding the heatmap image.
    """
    from .metrics import shape_volume_m3, simulation_window

    w, h = result.config.region_size_m
    t_lo, t_hi = simulation_window(result)
    grid = np.zeros((bins, bins))
    for intent in result.accepted:
        for v in intent.volumes or []:
            lo, hi = v.aabb()
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            bx = int(np.clip(cx / w * bins, 0, bins - 1))
            by = int(np.clip(cy / h * bins, 0, bins - 1))
            dur = max(0.0, min(v.t_end, t_hi) - max(v.t_start, t_lo))
            grid[by, bx] += shape_volume_m3(v.shape) * dur
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, origin="lower", extent=[0, w, 0, h], cmap="magma", aspect="equal")
    fig.colorbar(im, ax=ax, label="reserved volume-seconds (m³·s)")
    ax.set_title("Airspace congestion (reserved volume-seconds)")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    if out:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return ax


def delay_histogram(values, ax=None, out=None, bins=20, title="Delay distribution",
                    xlabel="total delay (s)", unit=" s", meanfmt=".0f",
                    overlay=None, labels=("all flights", "steady-state")):
    """Histogram of delay — how many flights suffered how much congestion lateness.

    ``xlabel`` / ``unit`` / ``meanfmt`` let the one plotter serve absolute seconds, the
    percent-of-trip flavour (:func:`delay_pct_histogram`), and the unbounded trip-time-inflation
    ratio (:func:`trip_ratio_histogram`). NaN (denied) flights are dropped.

    Parameters
    ------------
    - values (Sequence[float]): per-flight delays; NaN entries are dropped.
    - ax (Axes | None): target axes; a new figure is made when None.
    - out (path | None): if set, the figure is saved here (120 dpi) and closed.
    - bins (int | array): bin count or explicit edges passed to ``ax.hist``.
    - title (str): plot title.
    - xlabel (str): x-axis label.
    - unit (str): unit suffix appended to the mean annotation.
    - meanfmt (str): format spec for the mean value.
    - overlay (Sequence[float] | None): a second series drawn on shared bins beside ``values``, to
      compare the whole-run distribution against its steady-state-windowed twin.
    - labels (tuple[str, str]): names for the ``values`` and ``overlay`` series.

    Return
    --------
    - output (Axes): the axes the histogram was drawn on.
    """
    vals = np.asarray([v for v in values if v == v], float)  # drop NaN (denied) flights
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4.5))
    if overlay is None:
        ax.hist(vals, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
        mean = float(vals.mean()) if len(vals) else 0.0
        ax.axvline(mean, color="#dc2626", linestyle="--", lw=1.2, label=f"mean = {mean:{meanfmt}}{unit}")
        ax.set_title(f"{title}   (n={len(vals)})")
    else:
        ovals = np.asarray([v for v in overlay if v == v], float)
        # shared bin edges so the two distributions are directly comparable
        if np.isscalar(bins):
            pool = np.concatenate([vals, ovals]) if len(vals) or len(ovals) else np.array([0.0, 1.0])
            bins = np.histogram_bin_edges(pool, bins=bins)
        for series, color, lab in ((vals, "#2563eb", labels[0]), (ovals, "#f59e0b", labels[1])):
            m = float(series.mean()) if len(series) else 0.0
            ax.hist(series, bins=bins, color=color, alpha=0.55, edgecolor="white", linewidth=0.5,
                    label=f"{lab}  (n={len(series)}, mean {m:{meanfmt}}{unit})")
            ax.axvline(m, color=color, linestyle="--", lw=1.2)
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("flights")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    if out:
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def delay_histograms_by_lambda(per_flight_df, out=None, col="total_delay_s", xlabel="total delay (s)",
                               unit="s", bins=None, suptitle="Total-delay distribution by offered demand"):
    """One delay histogram per λ (shared x-axis) — the congestion distribution shifting with demand.

    ``per_flight_df`` is a concat of :func:`metrics.flight_frame` results, each tagged with its
    ``lam_per_hour``. Denied flights (NaN delay) are dropped; the surviving count is in each title.

    Parameters
    ------------
    - per_flight_df (DataFrame): per-flight rows carrying ``lam_per_hour`` and ``col``.
    - out (path | None): if set, the figure is saved here (120 dpi) and closed.
    - col (str): delay column to histogram (absolute-seconds or percent-of-trip flavour).
    - xlabel (str): x-axis label.
    - unit (str): unit suffix appended to each panel's mean annotation.
    - bins (int | array | None): shared bins; None spans 0 to the column max in 25 steps.
    - suptitle (str): figure super-title.

    Return
    --------
    - output (Figure): the figure of one panel per λ; raises ``ValueError`` if no flights survive.
    """
    df = per_flight_df.dropna(subset=[col])
    lams = sorted(df["lam_per_hour"].unique())
    if not lams:
        raise ValueError("no accepted flights to plot")
    if bins is None:
        bins = np.linspace(0, float(df[col].max()) or 1.0, 25)
    cols = min(3, len(lams))
    rows = (len(lams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                             sharex=True, squeeze=False)
    flat = axes.flatten()
    for ax, lam in zip(flat, lams):
        vals = df.loc[df["lam_per_hour"] == lam, col].to_numpy()
        ax.hist(vals, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="#dc2626", linestyle="--", lw=1.0,
                   label=f"mean={vals.mean():.0f}{unit}")
        ax.set_title(f"λ={lam:g}/h  (n={len(vals)})", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("flights")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    for ax in flat[len(lams):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return fig


def delay_pct_histogram(values, out=None, title="Delay as % of flight time", overlay=None):
    """Histogram of delay as a percentage of total trip time (bounded [0, 100)). ``overlay`` draws the
    steady-state-windowed twin alongside (see :func:`delay_histogram`)."""
    return delay_histogram(values, out=out, title=title, bins=np.linspace(0, 100, 21),
                           xlabel="delay (% of flight time)", unit="%", overlay=overlay)


def trip_ratio_histogram(values, out=None, title="Trip-time inflation", overlay=None):
    """Histogram of (straight-line flight time + delay) / straight-line flight time — the trip-time
    inflation factor. Unbounded (≥ 1): 1.0 = flew the ideal with no wait, 2.0 = took twice as long.
    The unbounded complement of :func:`delay_pct_histogram` (auto-ranged so the tail is visible).
    ``overlay`` draws the steady-state-windowed twin alongside."""
    return delay_histogram(values, out=out, title=title, bins=25,
                           xlabel="trip time ÷ straight-line time  (≥ 1, unbounded)", unit="×",
                           meanfmt=".2f", overlay=overlay)


def delay_pct_histograms_by_lambda(per_flight_df, out=None):
    """Per-λ histograms of delay as % of trip time — comparable across runs with different trip lengths."""
    return delay_histograms_by_lambda(
        per_flight_df, out=out, col="delay_pct", xlabel="delay (% of flight time)", unit="%",
        bins=np.linspace(0, 100, 21), suptitle="Delay-as-%-of-flight-time distribution by offered demand")


# The five levers of total_delay_s, in stack order: (key, label, colour, hatch). They sum exactly
# to total_delay_s, so the stack must carry all five — a missing lever silently under-reports the
# total. The detour is split into traffic vs lattice because on A*'s hex lattice most of it is
# quantization, not congestion (see context/figures/hex_lattice_overhead.png); the hatch marks
# that non-congestion band.
_DELAY_SOURCES = [
    ("ground_delay_s", "ground delay", "#2563eb", None),          # waited on the pad (FCFS queueing)
    ("air_hold_s", "air hold", "#f59e0b", None),                  # loitered/hovered mid-route
    ("detour_traffic_s", "detour — traffic", "#10b981", None),    # berth forced by other flights
    ("detour_lattice_s", "detour — lattice", "#9ca3af", "////"),  # hex quantization (see delay_sources)
    ("altitude_delay_phys_s", "climb — traffic-forced", "#6366f1", None),   # excess_m / climb_rate
]


def delay_sources(per_flight_df, out=None, by="lam_per_hour"):
    """Where delay comes from: stacked mean delay by source, over the five levers of ``total_delay_s``
    (ground hold / air hold / detour-traffic / detour-lattice / traffic-forced climb).

    Left panel = absolute seconds (how the total grows); right panel = % share (how the *mix* shifts —
    e.g. detour-dominated when sparse, ground-delay-dominated once the airspace saturates). The five
    sources sum exactly to ``total_delay_s`` (see :func:`metrics.delay_breakdown_s`).

    **Reading the hatched "detour — lattice" band.** It is real flight time, but for bare A* it is not
    caused by other traffic: a 6-direction lattice cannot fly the Euclidean straight line
    ``air_detour_m`` is measured against, so it stays pinned at the same value across every offered
    load. Ignore it when asking "how hard is traffic pushing flights sideways?" — that is the green
    band. Caveat for ``astar_shortcut``: there the refiner splices most of the staircase out, and what
    survives *does* grow with load (heavier traffic blocks more splices), so under a refiner the band
    is partly congestion-driven rather than pure geometry.

    Parameters
    ------------
    - per_flight_df (DataFrame): per-flight breakdown; filtered to accepted rows if an ``accepted``
      column is present, then NaN ``total_delay_s`` rows are dropped.
    - out (path | None): if set, the figure is saved here (120 dpi) and closed.
    - by (str): column to group bars on (λ); a single aggregate bar when absent or falsy.

    Return
    --------
    - output (Figure): the two-panel (absolute + share) figure.
    """
    df = per_flight_df
    if "accepted" in df.columns:
        df = df[df["accepted"]]
    df = df.dropna(subset=["total_delay_s"])
    # Legacy frames carry detour_time_s but not its two sub-bands. Back-fill as load_run does —
    # with no lattice info, the whole detour is traffic-attributable. This still partitions
    # detour_time_s exactly, so the five bands keep reconciling to total_delay_s.
    if "detour_traffic_s" not in df.columns:
        df = df.assign(detour_traffic_s=df["detour_time_s"], detour_lattice_s=0.0)
    # A run older than the climb band lacks altitude_delay_phys_s; zero-fill JUST that one band
    # (its climb info isn't recoverable from the parquet). Deliberately NOT a blanket fill over
    # _DELAY_SOURCES: every planner emits all five bands (0 where a lever doesn't apply — e.g. the
    # A*-only lattice split is 0 for the continuous planners), so a current-schema frame missing a
    # CORE lever (ground_delay_s/air_hold_s) is a real upstream defect that must still surface as a
    # KeyError, not a silent 0 band under a wrong-but-reconciling total.
    if "altitude_delay_phys_s" not in df.columns:
        df = df.assign(altitude_delay_phys_s=0.0)
    if by and by in df.columns:
        groups = sorted(df[by].unique())
        labels = [f"{g:g}" for g in groups]
        means = {k: np.array([df.loc[df[by] == g, k].mean() for g in groups])
                 for k, _, _, _ in _DELAY_SOURCES}
        xlabel = "offered load λ (req/h)"
    else:
        labels = ["all flights"]
        means = {k: np.array([df[k].mean()]) for k, _, _, _ in _DELAY_SOURCES}
        xlabel = ""

    fig, (a_abs, a_pct) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    total = sum(means[k] for k, _, _, _ in _DELAY_SOURCES)
    total_safe = np.where(total == 0, 1.0, total)

    for ax, normalize in ((a_abs, False), (a_pct, True)):
        bottom = np.zeros(len(labels))
        for k, name, color, hatch in _DELAY_SOURCES:
            v = means[k] / total_safe * 100 if normalize else means[k]
            ax.bar(x, v, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.5,
                   hatch=hatch)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3, axis="y")
    a_abs.set_title("Mean delay by source")
    a_abs.set_ylabel("seconds")
    a_abs.legend()
    a_pct.set_title("Delay composition (share)")
    a_pct.set_ylabel("% of total delay")
    a_pct.set_ylim(0, 100)

    fig.suptitle("Where delay comes from", fontsize=13)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return fig


def scene_3d(result: SimResult, t: float | None = None):
    """Assemble accepted reservations as a ``trimesh.Scene`` (boxes + cylinders) for a true-3D view.

    Parameters
    ------------
    - result (SimResult): run whose accepted intents become meshes.
    - t (float | None): show only volumes active at ``t``; None shows every accepted volume.

    Return
    --------
    - output (trimesh.Scene): the assembled scene, one translucent mesh per volume.
    """
    import trimesh

    scene = trimesh.Scene()
    hues = result_uss_hues(result)
    for intent in result.accepted:
        r, g, b = flight_color_by_uss(intent.request.uss_id, intent.request.flight_id, hues)
        color = [int(r * 255), int(g * 255), int(b * 255), 110]
        vols = intent.volumes if t is None else _active(intent.volumes, t)
        for v in vols:
            if isinstance(v.shape, BoxSpec):
                mesh = trimesh.creation.box(extents=v.shape.extents)
                tf = np.eye(4)
                tf[:3, :3] = v.shape.rotation()
                tf[:3, 3] = v.shape.center
                mesh.apply_transform(tf)
            else:
                s = v.shape
                mesh = trimesh.creation.cylinder(radius=s.radius, height=s.z_hi - s.z_lo)
                mesh.apply_translation([s.cx, s.cy, (s.z_lo + s.z_hi) / 2])
            mesh.visual.face_colors = color
            scene.add_geometry(mesh)
    return scene
