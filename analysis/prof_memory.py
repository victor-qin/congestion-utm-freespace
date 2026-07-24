"""Phase A (issue #8 Track A, memory plan): *falsify or confirm* the cache-layout diagnosis.

The measured fact this exists to explain: a lone worker plans at sequential speed, but 8 concurrent
workers each plan ~1.75x slower. Three mechanisms could do that, and they need completely different
fixes, so guessing is expensive:

1. **DVFS / scheduling** — cores clock down (or land on E-cores) when many are busy. No layout work
   would help. Probe: ``l1`` (32 KB working set, pure ALU + L1 hits) must stay ~1.0x at 8 procs.
2. **Shared-L2 capacity** — this machine's P-cores share **12 MB of L2 per 4-core cluster**, so 8
   workers get ~6 MB each, not 12. Probe: ``l2`` (3 MB each; 4 of them exactly fill one cluster).
3. **DRAM bandwidth** — a hard bytes/sec ceiling. Probe: ``dram`` (512 MB, always off-chip).

``soa_vs_aos`` then measures the *fix* rather than the problem: it replicates the kernel's exact
``_probe`` + ``_relax`` access pattern (same Fibonacci multiplier, same linear probe, same cap) in
both layouts —

* **SoA** (today): ``g_key``/``g_gen``/``g_val``/``g_came``/``g_flag`` are five separate allocations,
  so one relaxation touches **5 cache lines = 640 B** to move 33 B of payload (5% line utilization).
* **AoS** (proposed): one ``(cap, 4) int64`` block, 32 B per slot, 4 slots per 128 B line, with a
  ``float64`` view aliasing the same buffer for the g-value column — **1 line per relaxation**.

Both run at 1 and N processes. The SoA/AoS ratio at N procs is the expected Phase B payoff; if it is
~1.0 the whole layout plan is wrong and should be abandoned here rather than after implementing it.

Usage:
    uv run python analysis/prof_memory.py                    # topology + all probes
    uv run python analysis/prof_memory.py --procs 1,2,4,8,16
    uv run python analysis/prof_memory.py --only soa_vs_aos --distinct 20000,60000,200000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import subprocess
import time

import numpy as np
from numba import njit

# Mirrors astar_kernel._MAGIC / _slot0 exactly — the whole point is to reproduce that probe sequence.
_MAGIC = np.uint64(0x9E3779B97F4A7C15)
_LCG_A = np.uint64(6364136223846793005)
_LCG_C = np.uint64(1442695040888963407)
_GEN_MASK = np.int64(~np.int64(1))              # Phase B packs the closed bit into gen's bit 0

_LINE = 128                                     # M1 cache line; verified by topology()


# --------------------------------------------------------------------------------------- topology


def topology() -> dict:
    """Read the P-core cluster geometry that makes concurrent capacity contention possible."""
    keys = ["machdep.cpu.brand_string", "hw.perflevel0.physicalcpu", "hw.perflevel0.cpusperl2",
            "hw.perflevel0.l1dcachesize", "hw.perflevel0.l2cachesize", "hw.perflevel1.physicalcpu",
            "hw.cachelinesize", "hw.pagesize", "hw.memsize"]
    out = {}
    for k in keys:
        try:
            out[k] = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True,
                                    check=True).stdout.strip()
        except Exception:
            out[k] = "?"
    return out


def print_topology(t: dict) -> None:
    l2 = int(t.get("hw.perflevel0.l2cachesize", 0) or 0)
    per = int(t.get("hw.perflevel0.cpusperl2", 0) or 0)
    pcpu = int(t.get("hw.perflevel0.physicalcpu", 0) or 0)
    print(f"  cpu            {t['machdep.cpu.brand_string']}")
    print(f"  P-cores        {pcpu} in {pcpu // per if per else '?'} clusters of {per}")
    print(f"  E-cores        {t['hw.perflevel1.physicalcpu']}")
    print(f"  L1d / P-core   {int(t['hw.perflevel0.l1dcachesize']) / 1024:.0f} KB")
    print(f"  L2 / CLUSTER   {l2 / 2**20:.0f} MB  (shared by {per} cores)")
    print(f"  line / page    {t['hw.cachelinesize']} B / {int(t['hw.pagesize']) / 1024:.0f} KB")


# ------------------------------------------------------------------- synthetic contention probes


@njit(cache=True, nogil=True)
def _k_l1(buf, iters):
    """ALU-bound, 32 KB resident: isolates frequency/scheduling from every memory effect."""
    acc = 0.0
    n = buf.shape[0]
    for _ in range(iters):
        for i in range(n):
            acc = acc * 1.0000001 + buf[i]
    return acc


@njit(cache=True, nogil=True)
def _k_rand(buf, n_access, seed):
    """Independent random gathers over ``buf`` (power-of-two length): capacity/bandwidth-bound."""
    m = np.uint64(buf.shape[0] - 1)
    x = np.uint64(seed)
    acc = 0.0
    for _ in range(n_access):
        x = x * _LCG_A + _LCG_C
        acc += buf[np.int64((x >> np.uint64(30)) & m)]
    return acc


def _setup_l1(rep: int) -> tuple:
    return (np.arange(4096, dtype=np.float64), 150_000)  # 32 KB, resident in every core's own L1


def _run_l1(st) -> None:
    _k_l1(st[0], st[1])


def _setup_l2(rep: int) -> tuple:
    # 4 MB each: one process fits inside the 12 MB cluster L2; four on a cluster (16 MB) cannot.
    return (np.ones(1 << 19, np.float64), 200_000_000, 12345 + rep)


def _setup_dram(rep: int) -> tuple:
    return (np.ones(1 << 26, np.float64), 150_000_000, 999 + rep)   # 512 MB — always off-chip


def _run_rand(st) -> None:
    _k_rand(st[0], st[1], st[2])


# -------------------------------------------------------------- the layout experiment (Phase B)


@njit(cache=True, nogil=True)
def _hash_soa(g_key, g_gen, g_val, g_came, g_flag, gen, log2cap, n_ops, distinct, seed):
    """Today's layout: five separate arrays -> 5 cache lines per relaxation."""
    cap = g_key.shape[0]
    mask = cap - 1
    x = np.uint64(seed)
    hits = 0
    for _ in range(n_ops):
        x = x * _LCG_A + _LCG_C
        key = np.int64((x >> np.uint64(20)) % np.uint64(distinct))
        i = np.int64((np.uint64(key) * _MAGIC) >> np.uint64(64 - log2cap))
        for _p in range(cap):
            if g_gen[i] != gen:                          # _probe: touches g_gen ...
                break
            if g_key[i] == key:                          # ... and g_key (2nd line)
                hits += 1
                break
            i = (i + 1) & mask
        g_key[i] = key                                   # _relax: g_val / g_came / g_flag = 3 more
        g_gen[i] = gen
        g_val[i] = 1.5
        g_came[i] = key
        g_flag[i] = 0
    return hits


@njit(cache=True, nogil=True)
def _hash_aos(gp, gpf, gen, log2cap, n_ops, distinct, seed):
    """Proposed: one 32 B record per slot (cols key | gen|closed | val | came) -> 1 line."""
    cap = gp.shape[0]
    mask = cap - 1
    x = np.uint64(seed)
    hits = 0
    for _ in range(n_ops):
        x = x * _LCG_A + _LCG_C
        key = np.int64((x >> np.uint64(20)) % np.uint64(distinct))
        i = np.int64((np.uint64(key) * _MAGIC) >> np.uint64(64 - log2cap))
        for _p in range(cap):
            if (gp[i, 1] & _GEN_MASK) != gen:            # both probe fields share one line
                break
            if gp[i, 0] == key:
                hits += 1
                break
            i = (i + 1) & mask
        gp[i, 0] = key
        gp[i, 1] = gen
        gpf[i, 2] = 1.5                                  # float view, same 32 B row
        gp[i, 3] = key
    return hits


def aligned_2d(rows: int, cols: int, dtype=np.int64, align: int = _LINE) -> np.ndarray:
    """A ``(rows, cols)`` array whose base address is ``align``-aligned.

    ``np.empty`` only promises 16-64 B; a 32 B record group that straddles a 128 B line would give
    back much of the win, so records are placed deliberately."""
    itemsize = np.dtype(dtype).itemsize
    nbytes = rows * cols * itemsize
    raw = np.empty(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes].view(dtype).reshape(rows, cols)


def _setup_soa(rep: int, log2cap: int, n_ops: int, distinct: int) -> tuple:
    cap = 1 << log2cap
    st = (np.empty(cap, np.int64), np.zeros(cap, np.int64), np.empty(cap, np.float64),
          np.empty(cap, np.int64), np.empty(cap, np.int8), log2cap, n_ops, distinct, 7 + rep)
    return st


def _run_soa(st) -> None:
    _hash_soa(st[0], st[1], st[2], st[3], st[4], 2, st[5], st[6], st[7], st[8])


def _setup_aos(rep: int, log2cap: int, n_ops: int, distinct: int) -> tuple:
    gp = aligned_2d(1 << log2cap, 4, np.int64)
    gp[:, 1] = 0
    return (gp, gp.view(np.float64), log2cap, n_ops, distinct, 7 + rep)


def _run_aos(st) -> None:
    _hash_aos(st[0], st[1], 2, st[2], st[3], st[4], st[5])


# ------------------------------------------------------------------------------- process harness

_SETUP = {"l1": _setup_l1, "l2": _setup_l2, "dram": _setup_dram,
          "soa": _setup_soa, "aos": _setup_aos}
_RUN = {"l1": _run_l1, "l2": _run_rand, "dram": _run_rand, "soa": _run_soa, "aos": _run_aos}

_BARRIER = None


def _init(barrier) -> None:
    global _BARRIER
    _BARRIER = barrier


def _child(payload):
    """Setup, warm, *then* sync — so no process is timed while its siblings are still importing
    numpy/numba, loading the numba disk cache, or first-touching their buffers. Without the barrier
    those one-off costs land inside short timed regions and masquerade as contention."""
    kind, rep, kwargs = payload
    st = _SETUP[kind](rep, **kwargs)
    run = _RUN[kind]
    run(st)                                              # warm: JIT cache load + page first-touch
    if _BARRIER is not None:
        _BARRIER.wait()
    t0 = time.perf_counter()
    run(st)
    return time.perf_counter() - t0


def fanout(kind: str, nprocs: int, **kwargs) -> list[float]:
    """Run ``kind`` in ``nprocs`` concurrent processes; return each one's elapsed seconds.

    Every process does the *same* work, so the slowdown vs ``nprocs=1`` is pure interference, and
    the max/min spread across processes is the E-core-placement tell (E-cores are ~1.6x slower).
    Even ``nprocs=1`` goes through a pool, so the 1-process baseline has identical overheads."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(nprocs)
    with ctx.Pool(nprocs, initializer=_init, initargs=(barrier,)) as pool:
        return pool.map(_child, [(kind, r, kwargs) for r in range(nprocs)], chunksize=1)


def scaling_table(kinds: list[str], procs: list[int], **kwargs) -> dict:
    """For each kind: median elapsed at each process count, normalized to the 1-process time."""
    res = {}
    for kind in kinds:
        base = None
        rows = []
        for n in procs:
            ts = fanout(kind, n, **kwargs)
            med = statistics.median(ts)
            base = base if base is not None else med
            rows.append((n, med, med / base, min(ts), max(ts)))
        res[kind] = rows
    return res


def print_scaling(res: dict) -> None:
    print(f"  {'probe':<6} {'procs':>5} {'median s':>9} {'slowdown':>9} {'spread(max/min)':>16}")
    for kind, rows in res.items():
        for n, med, ratio, lo, hi in rows:
            print(f"  {kind:<6} {n:>5} {med:>9.3f} {ratio:>8.2f}x {hi / lo:>15.2f}x")


# ------------------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", default="1,2,4,8", help="process counts to fan out over")
    ap.add_argument("--only", default="all", choices=["all", "synthetic", "soa_vs_aos"])
    ap.add_argument("--log2cap", type=int, default=21, help="hash slots (21 = production ceiling)")
    ap.add_argument("--ops", type=int, default=4_000_000, help="relaxations per process")
    ap.add_argument("--distinct", default="60000",
                    help="distinct keys = the search's touched-slot footprint (comma list)")
    args = ap.parse_args()
    procs = [int(p) for p in args.procs.split(",")]

    t = topology()
    print("\n=== topology ===")
    print_topology(t)

    if args.only in ("all", "synthetic"):
        print("\n=== synthetic contention probes ===")
        print("  l1 ~1.0x  => not DVFS/scheduling;  l2 >> 1.0x => shared-L2 capacity is the term")
        print_scaling(scaling_table(["l1", "l2", "dram"], procs))

    if args.only in ("all", "soa_vs_aos"):
        line_b = _LINE
        for distinct in [int(d) for d in args.distinct.split(",")]:
            soa_mb = 5 * distinct * line_b / 2**20
            aos_mb = distinct * line_b / 2**20
            print(f"\n=== g-hash layout: {distinct:,} distinct slots, cap=2^{args.log2cap} ===")
            print(f"  predicted lines touched: SoA {soa_mb:6.1f} MB   AoS {aos_mb:6.1f} MB"
                  f"   ({soa_mb / aos_mb:.1f}x)")
            kw = dict(log2cap=args.log2cap, n_ops=args.ops, distinct=distinct)
            print(f"  {'procs':>5} {'SoA s':>9} {'AoS s':>9} {'AoS win':>9} "
                  f"{'SoA infl':>9} {'AoS infl':>9}")
            s_base = a_base = None
            for n in procs:
                s = statistics.median(fanout("soa", n, **kw))
                a = statistics.median(fanout("aos", n, **kw))
                s_base = s_base if s_base is not None else s
                a_base = a_base if a_base is not None else a
                print(f"  {n:>5} {s:>9.3f} {a:>9.3f} {s / a:>8.2f}x "
                      f"{s / s_base:>8.2f}x {a / a_base:>8.2f}x")

    print("\nread: 'AoS win' at the highest process count is the expected Phase B payoff;")
    print("      'infl' columns are each layout's own concurrency tax (the 1.75x to be removed).\n")


if __name__ == "__main__":
    main()
