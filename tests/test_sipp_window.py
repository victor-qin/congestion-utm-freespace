"""Per-plan safe-interval window: does it reproduce the global pool it replaces?

`sipp_window.build_window_intervals` derives a plan-local free-interval chain from the A* claim
arena, replacing `CompiledOccupancy`'s globally-maintained pool. Phase 3 deletes that pool, so these
tests are the whole parity argument — written now, against the live pool, so that a failure is
a red test rather than a wrong schedule. Design record: `context/sipp_runtime_plan.md`.

**The oracle is `CompiledHexOccupancy.blocked_py`, deliberately not
`CompiledOccupancy.free_intervals_py`.** The obvious choice is the wrong one twice over: it lives on
the class Phase 3 deletes, so the gate would stop compiling exactly when it is needed; and the two
structures use different boxes (`CompiledOccupancy` margin 48, `CompiledHexOccupancy` margin 64), so
a window cell legal in the arena's box returns `None` from the pool's method. Building the expected
complement from `blocked_py` needs one structure, has no box mismatch, and checks the same fold the
kernel reads.
"""
import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.dss import DSS
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.planner import sipp_window as SW
from freespace_sim.planner.astar.compiled_hex_occupancy import (
    _FIELD_MASK,
    _S0_SHIFT,
    _SPAN_BITS,
    CompiledHexOccupancy,
)
from freespace_sim.planner.astar.planner import _absorb
from freespace_sim.planner.astar.window import W_Q0, W_Q1, W_R0, W_R1, W_S0, W_S1
from freespace_sim.planner.sipp import SafeIntervalIndex
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import Volume4D

pytest.importorskip("numba", reason="the per-plan interval window is a numba kernel")

CFG = SimConfig()


# --------------------------------------------------------------------------- harness

def _build(cocc, wbox, *, own_cells=(), gen=1, slots=None, scratch=None):
    """Run the builder over `wbox`, growing once on a reported shortfall (what the host will do).

    Returns `(iv_lo, iv_hi, iv_nxt, tail, ov_own_gen)`. `own_cells` are stamped with `gen`, which is
    exactly how `AStarPlanner._build_overlay` marks the flight's own column footprint.
    """
    ov = np.zeros(cocc.NC, np.int32)
    for c in own_cells:
        ov[c] = gen
    n = (wbox[W_Q1] - wbox[W_Q0] + 1) * (wbox[W_R1] - wbox[W_R0] + 1) * cocc.n_levels
    size = slots if slots is not None else n + 16
    for _ in range(4):
        iv_lo = np.zeros(size, np.int32)
        iv_hi = np.zeros(size, np.int32)
        iv_nxt = np.full(size, -1, np.int32)
        scr = np.zeros(size if scratch is None else scratch, np.int64)
        tail = SW.build_window_intervals(
            cocc._arena.arena, cocc._arena.start, cocc._arena.length, cocc.static_col, ov, gen,
            cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox,
            iv_lo, iv_hi, iv_nxt, scr, _S0_SHIFT, _SPAN_BITS, _FIELD_MASK)
        if tail >= 0:
            return iv_lo, iv_hi, iv_nxt, tail, ov
        size = -tail
    raise AssertionError("builder kept reporting a shortfall after growing")


def _chain(iv_lo, iv_hi, iv_nxt, wcell):
    """The cell's chain as the kernel walks it: head slot IS `wcell`, then `iv_nxt`. Degenerate
    slots (`lo > hi`) are dropped, which is what every reader does."""
    out, slot = [], int(wcell)
    while slot != -1:
        lo, hi = int(iv_lo[slot]), int(iv_hi[slot])
        if lo <= hi:
            out.append((lo, hi))
        slot = int(iv_nxt[slot])
    return out


def _expected(cocc, q, r, L, s0, s1, own_cells=None):
    """Free runs of `[s0, s1]` from the surviving point oracle — the complement of `blocked_py`."""
    out, lo = [], None
    for s in range(s0, s1 + 1):
        if cocc.blocked_py(q, r, L, s, own_cells):
            if lo is not None:
                out.append((lo, s - 1)); lo = None
        elif lo is None:
            lo = s
    if lo is not None:
        out.append((lo, s1))
    return out


def _wbox(cocc, q0, q1, r0, r1, s0, s1):
    wb = SW.empty_wbox()
    from freespace_sim.planner.astar.window import W_RSPAN, W_STEPS
    wb[W_Q0] = q0; wb[W_Q1] = q1; wb[W_R0] = r0; wb[W_R1] = r1
    wb[W_S0] = s0; wb[W_S1] = s1
    wb[W_RSPAN] = r1 - r0 + 1; wb[W_STEPS] = s1 - s0 + 1
    return wb


def _wcells(cocc, wbox):
    """(wcell, q, r, L) in the builder's own iteration order — q-major, then r, then level."""
    w = -1
    for q in range(int(wbox[W_Q0]), int(wbox[W_Q1]) + 1):
        for r in range(int(wbox[W_R0]), int(wbox[W_R1]) + 1):
            for L in range(cocc.n_levels):
                w += 1
                yield w, q, r, L


def _congested(lam=400.0, horizon=600.0):
    """A real committed schedule — the regime where cells actually fragment into many intervals."""
    spec = with_overrides(get_scenario("metro_uniform"), lam_per_hour=lam,
                          horizon_s=horizon, seed=0)
    cfg = spec.config()
    dm = spec.demand_model() or UniformPoissonDemand()   # metro_uniform carries no model of its own
    reqs = dm.generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    from freespace_sim.uss import USS

    usses = {u: USS(u, dss, cfg, get_planner("astar")) for u in sc.uss_ids}
    for ev in sc.events:
        usses[ev.request.uss_id].handle_request(ev.request)
    cocc = CompiledHexOccupancy(cfg)
    # Deliberately no `evict_before`: `CompiledHexOccupancy._add` clips `s_lo` to `evicted_before`
    # and `CompiledOccupancy._add` does not, so an evicting fixture would desync the two structures
    # this file exists to compare.
    _absorb(cocc, led)
    return cfg, led, cocc


# --------------------------------------------------------------------------- the parity gate

def test_window_intervals_match_the_claim_oracle():
    """Every in-window cell's chain equals the complement of `blocked_py` over the window's steps.

    Scoped to FOREIGN cells (no own stamp); own-column transparency is the stronger case and
    has its own test. Chains are compared as sorted `(lo, hi)` lists rather than slot by slot: slot
    order is storage, and asserting on it would pin an implementation detail the kernel never reads.
    """
    cfg, _led, cocc = _congested()
    # A window over cells that were actually claimed, so this is not a test about empty space.
    claimed = [k >> 1 for k in range(2 * cocc.NC) if cocc._arena.length[k]]
    assert len(claimed) > 200, f"fixture is not congested enough ({len(claimed)} claimed cells)"
    mid = claimed[len(claimed) // 2]
    qr, L0 = divmod(mid, cocc.n_levels)
    iq, ir = divmod(qr, cocc.rspan)
    q0, r0 = iq + cocc.qmin, ir + cocc.rmin
    s1 = min(cocc.MAXS, 400)
    wbox = _wbox(cocc, q0 - 6, q0 + 6, r0 - 6, r0 + 6, 0, s1)

    iv_lo, iv_hi, iv_nxt, tail, _ov = _build(cocc, wbox)
    n_nonempty = 0
    for w, q, r, L in _wcells(cocc, wbox):
        got = sorted(_chain(iv_lo, iv_hi, iv_nxt, w))
        want = _expected(cocc, q, r, L, 0, s1)
        assert got == want, f"interval mismatch at ({q},{r},{L}): {got} vs {want}"
        if want != [(0, s1)]:
            n_nonempty += 1
    assert n_nonempty > 10, f"window held only {n_nonempty} claimed cells — the check was vacuous"
    assert tail > 0


def test_window_intervals_are_ascending():
    """The chain must ascend, and this is the ONLY thing checking it.

    `sipp_kernel._search` abandons a chain walk on "the chain ascends, so a later interval starts
    later still" — twice (`if a - 1 > hi_c: break` in the lateral block, `if ap > hi_c - rsteps:
    break` in the rung block). An unordered chain does not crash; it silently drops legal successors
    and returns a worse plan. The arena's slabs ARE unordered, because removal is a swap-remove, so
    ordering is a property this builder has to create rather than inherit.

    Committing the spans out of order is the point: a builder that emitted claims in slab order
    would pass a test that committed them left to right.
    """
    cfg = SimConfig()
    cocc = CompiledHexOccupancy(cfg)
    led = ReservationLedger(cfg)
    # SAME geometry, so every wall lands on the SAME cells and they accumulate claims; different,
    # deliberately NON-MONOTONE time windows, so slab order is not step order. (Spacing the walls
    # apart in x instead gives one claim per cell and tests nothing.)
    for fid, t0 in enumerate([320.0, 80.0, 240.0, 160.0, 400.0], start=1):
        vol = Volume4D(box_from_segment(vec(1000, -300, 150), vec(1000, 300, 150), 40, 400),
                       t0, t0 + 25.0)
        led.commit(fid, [vol])
    _absorb(cocc, led)
    claimed = [k >> 1 for k in range(2 * cocc.NC) if cocc._arena.length[k] >= 3]
    assert claimed, "no cell collected 3+ claims; the fixture cannot exercise ordering"

    qr, L0 = divmod(claimed[0], cocc.n_levels)
    iq, ir = divmod(qr, cocc.rspan)
    q0, r0 = iq + cocc.qmin, ir + cocc.rmin
    wbox = _wbox(cocc, q0 - 2, q0 + 2, r0 - 2, r0 + 2, 0, min(cocc.MAXS, 300))
    iv_lo, iv_hi, iv_nxt, _tail, _ov = _build(cocc, wbox)
    n_multi = 0
    for w, _q, _r, _L in _wcells(cocc, wbox):
        ch = _chain(iv_lo, iv_hi, iv_nxt, w)
        for a, b in zip(ch, ch[1:]):
            assert a[0] < b[0], f"chain not ascending at wcell {w}: {ch}"
            # The producer's other invariant: adjacent free runs are separated by at least one
            # blocked step. Two runs that merely touch mean the merge sweep failed to coalesce.
            assert a[1] + 1 < b[0], f"adjacent runs not separated at wcell {w}: {ch}"
        if len(ch) > 1:
            n_multi += 1
    assert n_multi > 0, "no cell produced a multi-interval chain — ordering was never exercised"


def test_window_intervals_handle_the_static_wall():
    """An always-active hub wall is empty for a foreign flight and fully free for its owner.

    This is the case `CompiledOccupancy` could only express by writing the wall into the same array
    as commit-derived blocks — the trap that bit twice during #125 (a naive cell rebuild silently
    un-walled a hub; the re-wall branch then dropped other owners' claims and raised on their
    release). Here the wall lives in `static_col` and never touches a claim slab, so both bugs are
    unrepresentable rather than guarded.
    """
    cfg = SimConfig(terminal_airspace_always_active=True)
    cocc = CompiledHexOccupancy(cfg)
    led = ReservationLedger(cfg)
    led.subscribe_static(cocc._on_static)
    term = Terminal("hub#0", capacity=1, radius=200.0)
    led.register_static_terminal(vec(1000, 0, 0), term)
    walled = [c for c in range(cocc.NC) if cocc.static_col[c]]
    assert walled, "the static wall marked no cells"

    c = walled[len(walled) // 2]
    qr, L = divmod(c, cocc.n_levels)
    iq, ir = divmod(qr, cocc.rspan)
    q, r = iq + cocc.qmin, ir + cocc.rmin
    wbox = _wbox(cocc, q, q, r, r, 0, 200)

    iv_lo, iv_hi, iv_nxt, _t, _ov = _build(cocc, wbox)
    assert _chain(iv_lo, iv_hi, iv_nxt, L) == [], "foreign flight sees free steps inside a hub wall"

    iv_lo, iv_hi, iv_nxt, _t, _ov = _build(cocc, wbox, own_cells=[c])
    assert _chain(iv_lo, iv_hi, iv_nxt, L) == [(0, 200)], "the owning hub's wall is not transparent"


def test_window_intervals_reproduce_the_overlay():
    """An own lane cell's chain equals `SafeIntervalIndex.free_intervals` for the same flight.

    That method is what `_sbuild_overlay` built the kernel's own-lane overlay out of, and the whole
    reason `SafeIntervalIndex` was subscribed to the ledger. If the `ov_own_gen` branch reproduces
    it, the overlay AND the index's subscription can both go — which is why this test is the licence
    for Phase 3 deleting two structures rather than one.
    """
    cfg = SimConfig()
    cocc = CompiledHexOccupancy(cfg)
    sidx = SafeIntervalIndex(cfg)
    led = ReservationLedger(cfg)
    # A foreign hub column over the cells our flight will own, plus an ordinary corridor through
    # them: the own fold must drop the column and KEEP the corridor.
    other = get_planner("astar").plan(
        FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0,
                      origin_terminal=Terminal("hub#1", capacity=1, radius=180.0)), led, cfg)
    assert other.accepted
    led.commit(1, other.volumes)
    _absorb(cocc, led); _absorb(sidx, led)

    own_tids = frozenset({"hub#1"})
    col_cells = [c for c, owners in cocc.col_owners.items() if owners <= own_tids]
    assert col_cells, "fixture committed no column cells to be own-transparent about"
    c = col_cells[len(col_cells) // 2]
    qr, L = divmod(c, cocc.n_levels)
    iq, ir = divmod(qr, cocc.rspan)
    q, r = iq + cocc.qmin, ir + cocc.rmin
    s1 = min(cocc.MAXS, 300)
    wbox = _wbox(cocc, q, q, r, r, 0, s1)

    iv_lo, iv_hi, iv_nxt, _t, _ov = _build(cocc, wbox, own_cells=[c])
    got = _chain(iv_lo, iv_hi, iv_nxt, L)
    want = sidx.free_intervals(q, r, L, own_tids, 0, s1, True)
    assert got == want, f"own-cell chain differs from the overlay it replaces: {got} vs {want}"


def test_window_intervals_report_a_slot_shortfall():
    """An undersized buffer must report how big it needs to be and write NOTHING.

    The capacity pass runs first for exactly this reason — the same contract as
    `claim_arena.add_many`. A builder that filled what it could and then bailed would leave a chain
    truncated mid-cell, which reads as "this cell is free after step k": a conflicting plan, not a
    slow one.
    """
    cfg, _led, cocc = _congested()
    claimed = [k >> 1 for k in range(2 * cocc.NC) if cocc._arena.length[k]]
    mid = claimed[len(claimed) // 2]
    qr, _L0 = divmod(mid, cocc.n_levels)
    iq, ir = divmod(qr, cocc.rspan)
    q0, r0 = iq + cocc.qmin, ir + cocc.rmin
    wbox = _wbox(cocc, q0 - 4, q0 + 4, r0 - 4, r0 + 4, 0, min(cocc.MAXS, 400))

    n = (wbox[W_Q1] - wbox[W_Q0] + 1) * (wbox[W_R1] - wbox[W_R0] + 1) * cocc.n_levels
    iv_lo = np.full(n, 7, np.int32)                # sentinels: untouched by a refusing build
    iv_hi = np.full(n, 7, np.int32)
    iv_nxt = np.full(n, 7, np.int32)
    scr = np.zeros(n, np.int64)
    ov = np.zeros(cocc.NC, np.int32)
    tail = SW.build_window_intervals(
        cocc._arena.arena, cocc._arena.start, cocc._arena.length, cocc.static_col, ov, 1,
        cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox,
        iv_lo, iv_hi, iv_nxt, scr, _S0_SHIFT, _SPAN_BITS, _FIELD_MASK)
    assert tail < 0, "a window sized to the head slots alone should not have fitted its overflow"
    assert -tail > n, f"reported need {-tail} does not exceed the head-slot count {n}"
    assert np.all(iv_lo == 7) and np.all(iv_hi == 7) and np.all(iv_nxt == 7), \
        "a refused build wrote into the buffer anyway"

    # ...and growing to exactly what it asked for succeeds.
    iv_lo2, iv_hi2, iv_nxt2, tail2, _ = _build(cocc, wbox, slots=-tail)
    assert tail2 > 0
    for w, q, r, L in _wcells(cocc, wbox):
        assert sorted(_chain(iv_lo2, iv_hi2, iv_nxt2, w)) == \
            _expected(cocc, q, r, L, 0, int(wbox[W_S1]))


def test_window_bounds_spans_the_whole_search_horizon():
    """The step span is `[base, max_step]` exactly — no heuristic tail.

    A* clips steps to `base + n_gsteps + tail_steps` because a step outside its span costs one probe
    and raises the miss flag. A SIPP interval's `hi` answers "how long may I wait here", so a short
    span raises nothing: it shortens a wait and returns a feasible, worse plan. Anyone copying A*'s
    signature would reintroduce that, so pin it.
    """
    cfg, _led, cocc = _congested()
    wb = SW.empty_wbox()
    n = SW.window_bounds(cocc, wb, q_cells=(0, 4), r_cells=(0, 4), base=12, max_step=900,
                         lateral_margin=3)
    assert n > 0
    assert (int(wb[W_S0]), int(wb[W_S1])) == (12, 900)

    # Degenerate boxes report 0 rather than writing wbox — the caller reads `W_STEPS == 0`.
    wb2 = SW.empty_wbox()
    assert SW.window_bounds(cocc, wb2, q_cells=(0, 4), r_cells=(0, 4), base=900, max_step=12,
                            lateral_margin=3) == 0
    assert int(wb2[W_S1]) == 0, "a degenerate box wrote bounds anyway"
