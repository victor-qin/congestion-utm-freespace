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
    """Visually-distinct RGB for a flight, hue stepped by the golden ratio."""
    h = (flight_id * _GOLDEN) % 1.0
    return colorsys.hsv_to_rgb(h, 0.62, 0.95)


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


def snapshot(result: SimResult, t: float, ax=None, out=None):
    """Top-down view of every reservation active at time ``t`` + drone dots."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8, 8))
    for intent in result.accepted:
        col = flight_color(intent.request.flight_id)
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
    ax.set_title(f"t = {t:.0f} s   ·   active flights: "
                 f"{sum(bool(_active(i.volumes, t)) for i in result.accepted)}")
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


def delay_histogram(values, ax=None, out=None, bins=20, title="Delay distribution"):
    """Histogram of total-delay seconds — how many flights suffered how much congestion lateness."""
    vals = np.asarray([v for v in values if v == v], float)  # drop NaN (denied) flights
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    mean = float(vals.mean()) if len(vals) else 0.0
    ax.axvline(mean, color="#dc2626", linestyle="--", lw=1.2, label=f"mean = {mean:.0f} s")
    ax.set_title(f"{title}   (n={len(vals)})")
    ax.set_xlabel("total delay (s)")
    ax.set_ylabel("flights")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    if out:
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def delay_histograms_by_lambda(per_flight_df, out=None, col="total_delay_s"):
    """One delay histogram per λ (shared x-axis) — the congestion distribution shifting with demand.

    ``per_flight_df`` is a concat of `metrics.flight_frame` results, each tagged with its
    ``lam_per_hour``. Denied flights (NaN delay) are dropped; the surviving count is in each title.
    """
    df = per_flight_df.dropna(subset=[col])
    lams = sorted(df["lam_per_hour"].unique())
    if not lams:
        raise ValueError("no accepted flights to plot")
    hi = float(df[col].max()) or 1.0
    bins = np.linspace(0, hi, 25)
    cols = min(3, len(lams))
    rows = (len(lams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                             sharex=True, squeeze=False)
    flat = axes.flatten()
    for ax, lam in zip(flat, lams):
        vals = df.loc[df["lam_per_hour"] == lam, col].to_numpy()
        ax.hist(vals, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="#dc2626", linestyle="--", lw=1.0,
                   label=f"mean={vals.mean():.0f}s")
        ax.set_title(f"λ={lam:g}/h  (n={len(vals)})", fontsize=11)
        ax.set_xlabel("total delay (s)")
        ax.set_ylabel("flights")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    for ax in flat[len(lams):]:
        ax.set_visible(False)
    fig.suptitle("Total-delay distribution by offered demand", fontsize=13)
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
    for intent in result.accepted:
        r, g, b = flight_color(intent.request.flight_id)
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
