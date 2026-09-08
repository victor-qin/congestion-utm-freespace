"""Can a purely SPATIAL reshaping of the corridor box buy colgen a 3-period cell window?

Background.  ``derive_cell_window`` measures, for one centre-cell crossing, which capacity-row
periods the ledger's reservation volumes occupy.  With the shipped geometry it returns ``(-2, 1)``
-- a four-period footprint -- which forces ``revisit_depth = 3`` and the wide dominance key.

The question here is whether moving the *longitudinal extension* (today ``+-corridor_width/2`` at
both ends of every segment) from symmetric to leading-only, and/or lengthening it forward, shrinks
that footprint the way a leading-only TIME pad does.

Everything below is a 1-D along-lane model of a straight cruise: a hex cell occupies
``[120k - 60, 120k + 60]`` metres along the centre-to-centre axis (inradius is exactly 60 m at the
shipped 120 m pitch), and a reservation box occupies a metre-interval x a time-interval.  For a
straight lane that is an exact model of the longitudinal question; lateral spill into the two
flanking cells is identical under every design considered, so it cannot separate them.

The model is validated against the real ``derive_cell_window`` on the shipped design before any
variant is reported.

    uv run python analysis/probe_cell_box_designs.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT not in Path(freespace_sim.__file__).resolve().parents:
    raise SystemExit("loaded the wrong tree")

from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.planner.colgen.windows import derive_cell_window  # noqa: E402
from freespace_sim.volumes import corridor_segment_volume  # noqa: E402

Box = tuple[float, float, float, float]  # (s_lo, s_hi, t_lo, t_hi), metres and seconds

K_RANGE = range(-4, 5)  # enough neighbours that no design can reach past the ends


def measured_pads(cfg: SimConfig) -> tuple[float, float]:
    """``(trail, lead)`` seconds the SHIPPED builder pads a segment by, measured not assumed.

    The tree's filing is not a constant: it has been symmetric ``(buf, buf)`` and leading-only
    ``(0, buf)``.  Reading it off ``corridor_segment_volume`` keeps every model below honest about
    whichever one is checked out.
    """
    dt = cfg.dt_s
    v = corridor_segment_volume((0.0, 0.0, 0.0), 0.0, (cfg.corridor_segment_len_m, 0.0, 0.0), dt,
                                cfg)
    return -v.t_start + 0.0, v.t_end - dt        # + 0.0 normalises an IEEE -0.0 for printing


# --------------------------------------------------------------------------- box families
def hop_boxes(*, aft: float, fore: float, trail: float, lead: float, pitch: float, dt: float
              ) -> list[Box]:
    """Today's shape: one box per HOP, centre -> centre, live for the whole hop.

    ``aft`` extends behind the start, ``fore`` beyond the end (``volumes.corridor_segment_volume``
    uses ``aft == fore == corridor_width_m / 2``); ``trail`` / ``lead`` are the time pads at each
    end, which the shipped builder currently sets to ``(0, time_buffer_s)``.
    """
    return [
        (pitch * k - aft, pitch * (k + 1) + fore, dt * k - trail, dt * (k + 1) + lead)
        for k in K_RANGE
    ]


def fine_boxes(
    *, sub_len: float, ext: float, trail: float, lead: float, pitch: float, dt: float
) -> list[Box]:
    """Resampled sub-boxes of length ``sub_len`` tiling the lane, each extended by ``ext``.

    This is the practical form of the cell box: rather than demanding a zero longitudinal
    extension (which is what actually seals consecutive boxes together), keep the extension and
    shrink the box until the extension only leaks ``ext`` metres past the hex boundary instead of
    a whole cell.  ``sub_len`` must divide the inradius so sub-box joints land on both cell centres
    and cell boundaries; the refiner already resamples to <= ``corridor_segment_len_m``.
    """
    v_nom = pitch / dt
    n = int(round(pitch / sub_len)) * 4
    return [
        (
            sub_len * i - ext,
            sub_len * (i + 1) + ext,
            sub_len * i / v_nom - trail,
            sub_len * (i + 1) / v_nom + lead,
        )
        for i in range(-n, n)
    ]


def cell_boxes(*, ext: float, trail: float, lead: float, pitch: float, dt: float, r: float
               ) -> list[Box]:
    """The alternative: one box per CELL, cut at the hex boundary, live only while inside it.

    The drone is at cell ``k``'s centre at ``k*dt`` and inside the cell over ``[k*dt - dt/2,
    k*dt + dt/2]``; the box covers exactly that cell, extended by ``ext`` at each end.
    """
    return [
        (
            pitch * k - r - ext,
            pitch * k + r + ext,
            dt * k - dt / 2.0 - trail,
            dt * k + dt / 2.0 + lead,
        )
        for k in K_RANGE
    ]


# --------------------------------------------------------------------------- the measurement
def claimed_periods(
    boxes: list[Box], cell_lo: float, cell_hi: float, dt: float, phase: float = 0.0
) -> set[int]:
    """Periods claimed on the cell spanning ``[cell_lo, cell_hi)``.

    Row ``j`` is the half-open interval ``[(j + phase) * dt, (j + 1 + phase) * dt)``; ``phase = 0``
    is the shipped anchoring at ``k * dt`` and ``phase = -0.5`` shifts every row half a period, so
    rows run boundary-crossing to boundary-crossing rather than centre to centre.
    """
    touched: set[int] = set()
    for s_lo, s_hi, t_lo, t_hi in boxes:
        if not (s_lo < cell_hi and s_hi > cell_lo):
            continue
        for j in range(-16, 17):
            if (j + phase) * dt < t_hi and (j + 1 + phase) * dt > t_lo:
                touched.add(j)
    return touched


def window(touched: set[int]) -> tuple[int, int, int]:
    """Inclusive ``(lo, hi)`` plus width; raises if the footprint is not contiguous."""
    lo, hi = min(touched), max(touched)
    if touched != set(range(lo, hi + 1)):
        raise RuntimeError(f"non-contiguous footprint {sorted(touched)}")
    return lo, hi, hi - lo + 1


def headway_s(boxes: list[Box], cell_lo: float, cell_hi: float) -> float:
    """Continuous-time width of the claim -- the same-lane minimum headway the ledger enforces.

    Two flights whose visits to this cell are closer than this in time hold overlapping volumes,
    independent of how the capacity rows are drawn.
    """
    spans = [(t_lo, t_hi) for s_lo, s_hi, t_lo, t_hi in boxes if s_lo < cell_hi and s_hi > cell_lo]
    return max(t for _, t in spans) - min(t for t, _ in spans)


def margins_s(boxes: list[Box], cell_lo: float, cell_hi: float, dt: float, pitch: float
              ) -> tuple[float, float]:
    """Seconds of schedule slip the cell's claim absorbs, ``(early, late)``.

    The drone is inside the cell over ``[cell_lo/v, cell_hi/v]``; the claim runs wider than that at
    each end, and those two margins are what actually protects a flight that runs off its plan.
    """
    v = pitch / dt
    spans = [(t_lo, t_hi) for s_lo, s_hi, t_lo, t_hi in boxes if s_lo < cell_hi and s_hi > cell_lo]
    return cell_lo / v - min(t for t, _ in spans), max(t for _, t in spans) - cell_hi / v


def coverage_gap_m(boxes: list[Box], pitch: float, dt: float, samples: int = 4001) -> float:
    """Longest stretch of the drone's OWN trajectory that sits inside no live box, in metres.

    Retracting a box's longitudinal reach is exactly the edit that can leave an aircraft outside
    its own reservation, so every design is checked for this before its window is believed.
    """
    v = pitch / dt
    span = 4.0 * pitch
    worst = run = 0.0
    for i in range(samples):
        s = -2.0 * pitch + span * i / (samples - 1)
        t = s / v
        covered = any(
            s_lo <= s <= s_hi and t_lo <= t < t_hi for s_lo, s_hi, t_lo, t_hi in boxes
        )
        run = 0.0 if covered else run + span / (samples - 1)
        worst = max(worst, run)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=Path, default=REPO_ROOT / "analysis" / "cell_box_idea.png")
    args = ap.parse_args()

    cfg = SimConfig()
    dt = cfg.dt_s
    pitch = cfg.corridor_segment_len_m          # 120 m == nominal_speed * dt
    r = pitch / 2.0                             # hex inradius at this pitch, exactly 60 m
    ext = cfg.corridor_width_m / 2.0            # the shipped longitudinal extension, 30 m
    buf = cfg.time_buffer_s
    trail, lead = measured_pads(cfg)            # whatever filing is CHECKED OUT, not assumed

    print(f"dt={dt}s  pitch={pitch}m  inradius={r}m  longitudinal ext={ext}m  time_buffer={buf}s")
    print(f"this tree files segments with pad (trail {trail:.0f} s, lead {lead:.0f} s)")

    # -- validate the 1-D model against the real derivation before trusting any variant ----------
    real = derive_cell_window(cfg)
    modelled = window(claimed_periods(
        hop_boxes(aft=ext, fore=ext, trail=trail, lead=lead, pitch=pitch, dt=dt), -r, r, dt
    ))[:2]
    print(f"\nderive_cell_window(cfg) = {real}   1-D model = {modelled}   "
          f"{'MATCH' if real == modelled else 'MISMATCH -- model is wrong, stop here'}")
    if real != modelled:
        raise SystemExit(1)

    designs: list[tuple[str, list[Box], float]] = [
        ("THIS TREE  hop box, ext 30/30, pad (%.0f, %.0f)" % (trail, lead),
         hop_boxes(aft=ext, fore=ext, trail=trail, lead=lead, pitch=pitch, dt=dt), 0.0),
        ("LEGACY  hop box, ext 30/30, symmetric pad (4, 4)",
         hop_boxes(aft=ext, fore=ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("--- spatial variants, all on the LEGACY symmetric pad ---", None, 0.0),
        ("LITERAL IDEA  hop box, aft ext 0, fore ext 60 m",
         hop_boxes(aft=0.0, fore=2 * ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("  variant  hop box, aft ext 0, fore ext 30 m",
         hop_boxes(aft=0.0, fore=ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("  variant  hop box, aft ext 60 m, fore ext 0",
         hop_boxes(aft=2 * ext, fore=0.0, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("  variant  hop box, NO longitudinal extension at all",
         hop_boxes(aft=0.0, fore=0.0, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("SLID BACK  aft +60 / fore -60, hop time window unchanged",
         hop_boxes(aft=r, fore=-r, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("SLID FWD   aft -60 / fore +60, hop time window unchanged",
         hop_boxes(aft=-r, fore=r, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("SLID BACK  aft +90 / fore -30 (30 m of overlap restored)",
         hop_boxes(aft=r + ext, fore=-r + ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("--- cell boxes, cut at the hex boundary ---", None, 0.0),
        ("CELL BOX  ext 0, symmetric pad 4 s, rows on k*dt",
         cell_boxes(ext=0.0, trail=buf, lead=buf, pitch=pitch, dt=dt, r=r), 0.0),
        ("CELL BOX  ext 0, symmetric pad 4 s, rows on (k-0.5)*dt",
         cell_boxes(ext=0.0, trail=buf, lead=buf, pitch=pitch, dt=dt, r=r), -0.5),
        ("CELL BOX  ext 30 m leaks into the neighbour, pad 4 s, half-offset rows",
         cell_boxes(ext=ext, trail=buf, lead=buf, pitch=pitch, dt=dt, r=r), -0.5),
        ("CELL BOX  ext 0, symmetric pad 2 s, rows on k*dt",
         cell_boxes(ext=0.0, trail=2.0, lead=2.0, pitch=pitch, dt=dt, r=r), 0.0),
        ("--- sub-cell boxes: the extension leaks only ext/v seconds ---", None, 0.0),
        ("FINE  30 m sub-boxes, ext 30 m, pad 4 s, rows on k*dt",
         fine_boxes(sub_len=30.0, ext=ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0),
        ("FINE  30 m sub-boxes, ext 30 m, pad 3 s, half-offset rows",
         fine_boxes(sub_len=30.0, ext=ext, trail=3.0, lead=3.0, pitch=pitch, dt=dt), -0.5),
        ("FINE  30 m sub-boxes, ext 30 m, pad 2 s, half-offset rows",
         fine_boxes(sub_len=30.0, ext=ext, trail=2.0, lead=2.0, pitch=pitch, dt=dt), -0.5),
        ("FINE  30 m sub-boxes, ext 30 m, pad 1 s, rows on k*dt",
         fine_boxes(sub_len=30.0, ext=ext, trail=1.0, lead=1.0, pitch=pitch, dt=dt), 0.0),
    ]

    head = f"\n{'design':<62} {'window':>10} {'W':>3} {'headway':>9} {'early':>7} {'late':>6} {'gap':>7}"
    print(head)
    print("-" * (len(head) - 1))
    for name, boxes, phase in designs:
        if boxes is None:                       # section heading, not a design
            print(name)
            continue
        lo, hi, w = window(claimed_periods(boxes, -r, r, dt, phase))
        early, late = margins_s(boxes, -r, r, dt, pitch)
        gap = coverage_gap_m(boxes, pitch, dt)
        print(f"{name:<62} {f'({lo},{hi})':>10} {w:>3} {headway_s(boxes, -r, r):>7.1f} s "
              f"{early:>6.1f}s {late:>5.1f}s {gap:>5.0f} m")

    draw(args.figure, dt=dt, pitch=pitch, r=r, ext=ext, buf=buf)


# --------------------------------------------------------------------------- the picture
def draw(out: Path, *, dt: float, pitch: float, r: float, ext: float, buf: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    SURFACE, INK, INK2, GRID = "#faf9f5", "#141413", "#6b6a66", "#d8d6d0"
    BLUE, ORANGE, GREEN = "#4a7fb5", "#d97757", "#3f7d5a"

    panels = [
        ("A.  TODAY — one box per HOP",
         hop_boxes(aft=ext, fore=ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0,
         "each box spans two cells and stays live for the whole hop",
         "16 s  =  4 s pad  +  8 s of two-cell box  +  4 s pad"),
        ("B.  the literal idea — aft extension 0, fore extension 60 m",
         hop_boxes(aft=0.0, fore=2 * ext, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0,
         "still two cells per box, so both hops still claim C",
         "16 s  —  identical to A, measured, for every aft/fore split"),
        ("C.  slide the box BACK 60 m — aft +60, fore −60",
         hop_boxes(aft=r, fore=-r, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0,
         "the box now covers the cell the drone is LEAVING",
         "12 s  —  same hop time window as A, only the geometry moved"),
        ("D.  slide it FORWARD instead — aft −60, fore +60",
         hop_boxes(aft=-r, fore=r, trail=buf, lead=buf, pitch=pitch, dt=dt), 0.0,
         "same box, carrying the ARRIVING hop's window instead",
         "12 s  —  W=3 too, but the window lands on (-2,0)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8), facecolor=SURFACE)
    fig.suptitle(
        "Reaching a 3-period cell window with geometry instead of an asymmetric time pad\n"
        "(every panel holds the LEGACY symmetric ±4 s pad fixed, to isolate the geometry)\n"
        "space–time view of one lane — a reservation box is a rectangle, and a cell's capacity "
        "claim is the shadow every box touching it casts on the time axis",
        fontsize=13, color=INK, y=0.982,
    )

    s_lim, t_lim = 250.0, 12.5
    for ax, (title, boxes, phase, note, decomp) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        touched = claimed_periods(boxes, -r, r, dt, phase)
        lo_j, hi_j, w = window(touched)

        # the hex cells along the lane; C is the one whose rows we are counting
        for k in (-2, -1, 0, 1, 2):
            ax.axvline(pitch * k - r, color=GRID, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)
            ax.axvline(pitch * k + r, color=GRID, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)
        ax.axvspan(-r, r, facecolor=GREEN, alpha=0.07, zorder=0)
        ax.annotate("cell C", (0, t_lim - 0.5), fontsize=10, color=GREEN, ha="center", va="top",
                    weight="bold", zorder=8)

        # capacity rows at this panel's phase; the claimed ones are shaded
        for j in range(-6, 7):
            y0, y1 = (j + phase) * dt, (j + 1 + phase) * dt
            if not -t_lim < (y0 + y1) / 2 < t_lim:
                continue
            if j in touched:
                ax.axhspan(y0, y1, facecolor=GREEN, alpha=0.11, zorder=0)
            ax.axhline(y0, color=GRID, linewidth=1.0, zorder=1)
            ax.annotate(f"row {j:+d}".replace("+0", " 0"), (1.015, (y0 + y1) / 2),
                        xycoords=("axes fraction", "data"), fontsize=8,
                        color=GREEN if j in touched else INK2, ha="left", va="center",
                        weight="bold" if j in touched else "normal", annotation_clip=False)

        for s0, s1, t0, t1 in boxes:
            hits = s0 < r and s1 > -r
            ax.add_patch(Rectangle((s0, t0), s1 - s0, t1 - t0,
                                   facecolor=ORANGE if hits else BLUE,
                                   alpha=0.26 if hits else 0.11,
                                   edgecolor=ORANGE if hits else BLUE,
                                   linewidth=1.6 if hits else 0.8, zorder=3))

        # the drone: cell centres on the ticks, 30 m/s between; bold while inside cell C
        ax.plot([-s_lim, s_lim], [-s_lim / pitch * dt, s_lim / pitch * dt],
                color=INK2, linewidth=1.2, zorder=5)
        ax.plot([-r, r], [-r / pitch * dt, r / pitch * dt], color=INK, linewidth=3.2, zorder=6)
        ax.plot([0], [0], "o", color=INK, markersize=7, zorder=7)

        spans = [(t0, t1) for s0, s1, t0, t1 in boxes if s0 < r and s1 > -r]
        c0, c1 = min(t for t, _ in spans), max(t for _, t in spans)
        ax.annotate("", (-s_lim + 20, c0), (-s_lim + 20, c1),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=2.0), zorder=8)
        ax.annotate(f"cell C held\n{c1 - c0:.0f} s", (-s_lim + 28, (c0 + c1) / 2),
                    fontsize=9, color=ORANGE, ha="left", va="center", weight="bold", zorder=8)

        ax.set_xlim(-s_lim, s_lim)
        ax.set_ylim(-t_lim, t_lim)
        ax.set_xlabel("distance along the lane (m)", fontsize=9, color=INK2)
        ax.set_ylabel("time (s) — 0 = drone at cell C's centre", fontsize=9, color=INK2)
        early, late = margins_s(boxes, -r, r, dt, pitch)
        ax.set_title(
            f"{title}\nW = {w},  window {(lo_j, hi_j)} — {note}\n"
            f"{decomp}\nabsorbs {early:.0f} s early / {late:.0f} s late",
            fontsize=10.2, color=INK, loc="left", pad=9, linespacing=1.5,
        )
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)

    notes = [
        "Orange = a box that touches cell C and therefore claims its capacity rows;  blue = one that does not.  The drone's own trajectory is the diagonal.",
        "A hop box ends at cell C's centre and the next one starts from it, so BOTH always claim C — which is why B measures identically to A for every aft/fore",
        "split, including deleting the extension outright. Only a FULL half-pitch slide (C, D) leaves one claimant; it is the same box, differing in which hop's window it carries.",
        "Caveat: C and D make consecutive boxes ABUT rather than overlap. At a 4 s pad the 12 s claim budget is exactly spent, so restoring even 30 m of overlap costs a",
        "whole period (measured: aft +90 / fore −30 gives W=5). Keeping the overlap needs ~30 m sub-boxes, a 3 s pad and half-offset rows — that one is W=3 at ±4 s.",
    ]
    for i, line in enumerate(notes):
        fig.text(0.008, 0.058 - i * 0.0126, line, fontsize=8.6, color=INK if i == 0 else INK2)
    fig.subplots_adjust(left=0.055, right=0.935, top=0.855, bottom=0.140, hspace=0.42, wspace=0.20)
    fig.savefig(out, dpi=190, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
