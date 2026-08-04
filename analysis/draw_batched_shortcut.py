"""Render the astar_batched_shortcut algorithm explainer as a PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, RegularPolygon


OUT = Path(__file__).with_name("astar_batched_shortcut.png")

INK = "#18324A"
RAW = "#35566F"
MUTED = "#6B7C8B"
GRID = "#DDE5EA"
GHOST = "#B8C4CC"
GREEN = "#11875D"
GREEN_LIGHT = "#DDF3EA"
RED = "#C84A47"
RED_LIGHT = "#F9E4E2"
GOLD = "#D99000"
GOLD_LIGHT = "#FFF1CC"
PANEL = "#F7F9FA"
WHITE = "#FFFFFF"


POINTS = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (2.0, 0.0),
    "D": (3.0, 0.0),
    "E": (4.0, 0.0),
    "F": (5.0, 0.0),
    "G": (6.0, -1.0),
    "H": (7.0, -2.0),
    "I": (8.0, -3.0),
}
RAW_IDS = list(POINTS)


def _panel(ax, title: str, subtitle: str = "") -> None:
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(-3.75, 1.52)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (-0.52, -3.70), 9.02, 5.10,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=PANEL, edgecolor=GRID, linewidth=1.2, zorder=-10,
    ))
    ax.text(-0.25, 1.18, title, color=INK, fontsize=13.5, fontweight="bold", va="top")
    if subtitle:
        ax.text(-0.25, 0.77, subtitle, color=MUTED, fontsize=9.4, va="top")


def _xy(ids: list[str]) -> np.ndarray:
    return np.asarray([POINTS[name] for name in ids], dtype=float)


def _hexes(ax, alpha: float = 1.0) -> None:
    for x, y in POINTS.values():
        ax.add_patch(RegularPolygon(
            (x, y), numVertices=6, radius=0.34, orientation=np.pi / 6,
            facecolor=WHITE, edgecolor=GRID, linewidth=0.85, alpha=alpha, zorder=-2,
        ))


def _raw_route(ax, alpha: float = 0.34, labels: bool = True) -> None:
    raw = _xy(RAW_IDS)
    ax.plot(raw[:, 0], raw[:, 1], color=RAW, linewidth=2.2, alpha=alpha, zorder=0)
    ax.scatter(raw[:, 0], raw[:, 1], s=27, color=RAW, alpha=alpha, zorder=1)
    if labels:
        for name, (x, y) in POINTS.items():
            dy = 0.34 if name not in ("G", "H", "I") else -0.40
            ax.text(x, y + dy, name, ha="center", va="center", color=MUTED,
                    fontsize=9.3, fontweight="bold", zorder=5)


def _samples(ax, left: str, right: str, count: int = 5) -> None:
    a = np.asarray(POINTS[left])
    b = np.asarray(POINTS[right])
    ts = np.linspace(0.15, 0.85, count)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    ax.scatter(pts[:, 0], pts[:, 1], s=22, facecolor=WHITE, edgecolor=GREEN,
               linewidth=1.3, zorder=6)


def _candidate(ax, ids: list[str], chord: tuple[str, str]) -> None:
    route = _xy(ids)
    ax.plot(route[:, 0], route[:, 1], color=GREEN, linewidth=3.4,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(route[:, 0], route[:, 1], s=48, facecolor=WHITE, edgecolor=GREEN,
               linewidth=2.0, zorder=5)
    left, right = chord
    lx, ly = POINTS[left]
    rx, ry = POINTS[right]
    ax.plot([lx, rx], [ly, ry], color=GREEN, linewidth=4.8,
            solid_capstyle="round", zorder=4)
    _samples(ax, left, right)


def _step_panel(ax, title: str, subtitle: str, ids: list[str], chord: tuple[str, str],
                removed: list[str]) -> None:
    _panel(ax, title, subtitle)
    _hexes(ax)
    _raw_route(ax)
    _candidate(ax, ids, chord)
    for name in removed:
        x, y = POINTS[name]
        ax.scatter([x], [y], s=75, facecolor=WHITE, edgecolor=GHOST,
                   linewidth=1.5, zorder=6)
        ax.plot([x - 0.12, x + 0.12], [y - 0.12, y + 0.12], color=GHOST,
                linewidth=1.4, zorder=7)
        ax.plot([x - 0.12, x + 0.12], [y + 0.12, y - 0.12], color=GHOST,
                linewidth=1.4, zorder=7)


def _raw_panel(ax) -> None:
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-3.75, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _hexes(ax)
    raw = _xy(RAW_IDS)
    ax.plot(raw[:, 0], raw[:, 1], color=RAW, linewidth=4.0,
            solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(raw[:, 0], raw[:, 1], s=64, facecolor=WHITE, edgecolor=RAW,
               linewidth=2.1, zorder=4)
    for name, (x, y) in POINTS.items():
        dy = 0.40 if name not in ("G", "H", "I") else -0.45
        ax.text(x, y + dy, name, ha="center", color=INK, fontsize=11,
                fontweight="bold")
    fx, fy = POINTS["F"]
    ax.scatter([fx], [fy], s=185, facecolor=GOLD_LIGHT, edgecolor=GOLD,
               linewidth=2.5, zorder=3)
    ax.scatter([fx], [fy], s=52, facecolor=WHITE, edgecolor=RAW,
               linewidth=2.0, zorder=4)
    ax.annotate(
        "3D heading changes here",
        xy=(fx, fy), xytext=(5.25, 0.78),
        color=INK, fontsize=10.5, ha="left",
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.7},
    )
    ax.text(2.28, -0.73, "incoming straight run  A … E → F", color=RAW,
            fontsize=10.2, fontweight="bold", ha="center")
    ax.text(7.00, -3.56, "outgoing straight run  F → G … I", color=RAW,
            fontsize=10.2, fontweight="bold", ha="center")


def _fallback_panel(ax) -> None:
    ax.set_xlim(-0.35, 6.55)
    ax.set_ylim(-0.8, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (-0.3, -0.72), 6.78, 4.72,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=PANEL, edgecolor=GRID, linewidth=1.2, zorder=-10,
    ))
    ax.text(-0.05, 3.72, "Fallback keeps going after a failure", color=INK,
            fontsize=13.5, fontweight="bold", va="top")
    ax.text(-0.05, 3.34, "Feasibility is not monotone with chord length or retiming.",
            color=MUTED, fontsize=9.5, va="top")

    anchors = {"B": (0.6, 0.45), "C": (1.65, 0.45), "D": (2.7, 0.45), "A": (-0.05, 0.45)}
    g = (5.65, 2.25)
    for name, (x, y) in anchors.items():
        ax.scatter([x], [y], s=42, facecolor=WHITE, edgecolor=RAW, linewidth=1.7, zorder=4)
        ax.text(x, y - 0.38, name, ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.scatter([g[0]], [g[1]], s=52, facecolor=WHITE, edgecolor=RAW, linewidth=1.8, zorder=4)
    ax.text(g[0] + 0.18, g[1] + 0.18, "G", color=INK, fontsize=9.5, fontweight="bold")

    # Maximal and first fallback fail; the scan still reaches and accepts C→G.
    ax.plot([anchors["A"][0], g[0]], [anchors["A"][1], g[1]], color=RED,
            linewidth=2.3, linestyle=(0, (5, 4)), solid_capstyle="round")
    ax.plot([anchors["D"][0], g[0]], [anchors["D"][1], g[1]], color=RED,
            linewidth=2.3, linestyle=(0, (5, 4)), solid_capstyle="round")
    ax.plot([anchors["C"][0], g[0]], [anchors["C"][1], g[1]], color=GREEN,
            linewidth=2.5, solid_capstyle="round")
    ax.plot([anchors["B"][0], g[0]], [anchors["B"][1], g[1]], color=GHOST,
            linewidth=1.7, linestyle=(0, (2, 3)), solid_capstyle="round")
    ax.text(1.15, 2.57, "A→G  ✕ maximal probe", color=RED, fontsize=9.8, fontweight="bold")
    ax.text(3.48, 1.70, "D→G  ✕", color=RED, fontsize=9.6, fontweight="bold")
    ax.text(2.82, 0.92, "C→G  ✓", color=GREEN, fontsize=9.6, fontweight="bold")
    ax.text(-0.02, -0.30, "If A→G fails: try D, C, B in order — a rejection never prunes later anchors.",
            color=INK, fontsize=9.3)


def _semantics_panel(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.04), 9.94, 5.9,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=PANEL, edgecolor=GRID, linewidth=1.2, zorder=-10,
    ))
    ax.text(0.42, 5.45, "Logical chord ≠ reservation sampling", color=INK,
            fontsize=13.5, fontweight="bold", va="top")
    ax.text(0.42, 5.02, "The shortcut state gets fewer corners; the filed intent stays finely segmented.",
            color=MUTED, fontsize=9.5, va="top")

    ax.text(0.48, 4.18, "logical state", color=MUTED, fontsize=9.2, fontweight="bold")
    ax.plot([2.55, 8.7], [4.25, 4.25], color=GREEN, linewidth=4.2, solid_capstyle="round")
    ax.scatter([2.55, 8.7], [4.25, 4.25], s=62, facecolor=WHITE, edgecolor=GREEN,
               linewidth=2.0, zorder=3)
    ax.text(2.55, 4.58, "A", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(8.7, 4.58, "I", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(5.63, 4.58, "one literal A→I edge", ha="center", color=GREEN,
            fontsize=9.5, fontweight="bold")

    ax.text(0.48, 3.08, "filed centerline", color=MUTED, fontsize=9.2, fontweight="bold")
    xs = np.linspace(2.55, 8.7, 9)
    ax.plot(xs, np.full_like(xs, 3.15), color=RAW, linewidth=2.2)
    ax.scatter(xs, np.full_like(xs, 3.15), s=34, facecolor=WHITE, edgecolor=RAW,
               linewidth=1.4, zorder=3)
    ax.text(5.63, 2.69, "collinear samples ≤ corridor_segment_len_m", ha="center",
            color=INK, fontsize=9.4)

    ax.text(0.48, 1.91, "every candidate", color=MUTED, fontsize=9.2, fontweight="bold")
    stages = ["rebuild", "resample", "retime", "detour", "ledger", "terminal capacity"]
    widths = [0.92, 1.05, 0.86, 0.86, 0.82, 1.62]
    x = 2.20
    for index, (label, width) in enumerate(zip(stages, widths, strict=True)):
        face = GREEN_LIGHT if index == len(stages) - 1 else WHITE
        ax.add_patch(FancyBboxPatch(
            (x, 1.53), width, 0.57,
            boxstyle="round,pad=0.02,rounding_size=0.11",
            facecolor=face, edgecolor=GRID, linewidth=1.0,
        ))
        ax.text(x + width / 2, 1.815, label, ha="center", va="center",
                color=INK, fontsize=8.5, fontweight="bold")
        if index < len(stages) - 1:
            ax.text(x + width + 0.10, 1.815, "→", ha="center", va="center",
                    color=MUTED, fontsize=10.5)
        x += width + 0.25

    ax.add_patch(FancyBboxPatch(
        (0.44, 0.47), 9.06, 0.61,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=GOLD_LIGHT, edgecolor="#E7C76F", linewidth=1.0,
    ))
    ax.text(4.97, 0.77,
            "Fast path: 3 full probes (E→G, A→G, A→I), independent of straight-run length",
            ha="center", va="center", color=INK, fontsize=9.3, fontweight="bold")


def main() -> None:
    fig = plt.figure(figsize=(16, 11), facecolor=WHITE)
    fig.text(0.055, 0.965, "A* batched shortcut: back a corner cut across straight runs",
             color=INK, fontsize=24, fontweight="bold", va="top")
    fig.text(0.055, 0.930,
             "Detect the real 3D turn once, then probe the local cut and both maximal run endpoints.",
             color=MUTED, fontsize=12.5, va="top")

    raw_ax = fig.add_axes((0.18, 0.665, 0.64, 0.245))
    _raw_panel(raw_ax)

    ax1 = fig.add_axes((0.045, 0.380, 0.292, 0.255))
    ax2 = fig.add_axes((0.354, 0.380, 0.292, 0.255))
    ax3 = fig.add_axes((0.663, 0.380, 0.292, 0.255))
    _step_panel(
        ax1, "1  Local seed  E→G", "Cheap corner probe; F is removable.",
        ["A", "B", "C", "D", "E", "G", "H", "I"], ("E", "G"), ["F"],
    )
    _step_panel(
        ax2, "2  Incoming batch  A→G", "Jump directly to the run start.",
        ["A", "G", "H", "I"], ("A", "G"), ["B", "C", "D", "E", "F"],
    )
    _step_panel(
        ax3, "3  Outgoing batch  A→I", "Back the accepted cut through the exit run.",
        ["A", "I"], ("A", "I"), ["B", "C", "D", "E", "F", "G", "H"],
    )

    fallback_ax = fig.add_axes((0.047, 0.045, 0.405, 0.295))
    semantics_ax = fig.add_axes((0.477, 0.045, 0.478, 0.295))
    _fallback_panel(fallback_ax)
    _semantics_panel(semantics_ax)

    fig.savefig(OUT, dpi=160, facecolor=WHITE, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
