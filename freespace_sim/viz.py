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

    Pass ``uss`` to slice the view to a single operator's flights.
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
    busiest). The free-space analog of the sibling's hex-occupancy heatmap."""
    from .metrics import shape_volume_m3

    w, h = result.config.region_size_m
    grid = np.zeros((bins, bins))
    for intent in result.accepted:
        for v in intent.volumes or []:
            lo, hi = v.aabb()
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            bx = int(np.clip(cx / w * bins, 0, bins - 1))
            by = int(np.clip(cy / h * bins, 0, bins - 1))
            dur = max(0.0, min(v.t_end, result.config.horizon_s) - max(v.t_start, 0.0))
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
                    xlabel="total delay (s)", unit=" s"):
    """Histogram of delay — how many flights suffered how much congestion lateness.

    ``xlabel``/``unit`` let the same plotter serve both absolute seconds and the percent-of-trip
    flavour (see :func:`delay_pct_histogram`). NaN (denied) flights are dropped.
    """
    vals = np.asarray([v for v in values if v == v], float)  # drop NaN (denied) flights
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    mean = float(vals.mean()) if len(vals) else 0.0
    ax.axvline(mean, color="#dc2626", linestyle="--", lw=1.2, label=f"mean = {mean:.0f}{unit}")
    ax.set_title(f"{title}   (n={len(vals)})")
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

    ``per_flight_df`` is a concat of `metrics.flight_frame` results, each tagged with its
    ``lam_per_hour``. Denied flights (NaN delay) are dropped; the surviving count is in each title.
    ``col``/``xlabel``/``unit``/``bins`` select the absolute-seconds or percent-of-trip flavour.
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


def delay_pct_histogram(values, out=None, title="Delay as % of flight time"):
    """Histogram of delay as a percentage of total trip time (bounded [0, 100))."""
    return delay_histogram(values, out=out, title=title, bins=np.linspace(0, 100, 21),
                           xlabel="delay (% of flight time)", unit="%")


def delay_pct_histograms_by_lambda(per_flight_df, out=None):
    """Per-λ histograms of delay as % of trip time — comparable across runs with different trip lengths."""
    return delay_histograms_by_lambda(
        per_flight_df, out=out, col="delay_pct", xlabel="delay (% of flight time)", unit="%",
        bins=np.linspace(0, 100, 21), suptitle="Delay-as-%-of-flight-time distribution by offered demand")


_DELAY_SOURCES = [
    ("ground_delay_s", "ground delay", "#2563eb"),   # waited on the pad (FCFS queueing)
    ("air_hold_s", "air hold", "#f59e0b"),            # loitered/hovered mid-route
    ("detour_time_s", "detour time", "#10b981"),      # extra path length, as lateness-seconds
]


def delay_sources(per_flight_df, out=None, by="lam_per_hour"):
    """Where delay comes from: stacked mean delay by source (ground / air-hold / detour-time).

    Left panel = absolute seconds (how the total grows); right panel = % share (how the *mix*
    shifts — e.g. detour-dominated when sparse, ground-delay-dominated once the airspace saturates).
    Groups by ``by`` (λ) if that column is present, else shows a single aggregate bar. The three
    sources sum exactly to ``total_delay_s``.
    """
    df = per_flight_df
    if "accepted" in df.columns:
        df = df[df["accepted"]]
    df = df.dropna(subset=["total_delay_s"])
    if by and by in df.columns:
        groups = sorted(df[by].unique())
        labels = [f"{g:g}" for g in groups]
        means = {k: np.array([df.loc[df[by] == g, k].mean() for g in groups]) for k, _, _ in _DELAY_SOURCES}
        xlabel = "offered load λ (req/h)"
    else:
        labels = ["all flights"]
        means = {k: np.array([df[k].mean()]) for k, _, _ in _DELAY_SOURCES}
        xlabel = ""

    fig, (a_abs, a_pct) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    total = sum(means[k] for k, _, _ in _DELAY_SOURCES)
    total_safe = np.where(total == 0, 1.0, total)

    for ax, normalize in ((a_abs, False), (a_pct, True)):
        bottom = np.zeros(len(labels))
        for k, name, color in _DELAY_SOURCES:
            v = means[k] / total_safe * 100 if normalize else means[k]
            ax.bar(x, v, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.5)
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
    """Assemble active reservations as a `trimesh.Scene` (boxes + cylinders) for a true-3D view.

    With ``t=None`` every accepted volume is shown; otherwise only those active at ``t``.
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
