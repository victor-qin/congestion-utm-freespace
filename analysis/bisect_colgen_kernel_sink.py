"""Find the exact hop where a reference sink's path dies inside the kernel."""
import random

import numpy as np

import freespace_sim

assert "/vientiane/" in freespace_sim.__file__

from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.planner import hexgrid as hg  # noqa: E402
from freespace_sim.planner.colgen import dp_kernel, dp_prepare, pricing  # noqa: E402
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph  # noqa: E402
from freespace_sim.planner.colgen.objective import cost_model  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.pricing import DualView  # noqa: E402
from freespace_sim.types import FlightRequest, Terminal, vec  # noqa: E402

cfg = SimConfig(planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                region_size_m=(20000.0, 20000.0), terminal_airspace_always_active=True,
                max_ground_delay_s=48.0, max_detour_factor=10.0)


def point(c):
    x, y = hg.hex_center(*c, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


o, d = point((0, 0)), point((4, -1))
ot, dt = Terminal("kern-A", 1, radius=90.0), Terminal("kern-B", 1, radius=90.0)
params = ColGenParams(solver="highs", max_air_overrun_hops=4)
fg = build_flight_graph(
    FlightRequest(12, o, d, 0.0, 0.0, origin_terminal=ot, dest_terminal=dt),
    cfg, [(o, ot), (d, dt)], params,
)
model = cost_model(params, cfg)
rng = random.Random(606)
duals = {}
for cell in list(fg.corridor_cells)[:40]:
    for step in range(fg.min_step, min(fg.min_step + 25, fg.max_step + 1)):
        if rng.random() < 0.55:
            duals[RowKey.cell(cell[0], cell[1], 0, step)] = rng.uniform(-2.0, 40.0) * (
                10.0 ** rng.randint(-6, 3)
            )
view = DualView(duals, cfg)

topo = dp_prepare.prepare_topology(fg, cfg)
rows = dp_prepare.prepare_rows(fg, cfg, topo)
pd = dp_prepare.prepare_duals(view, fg, topo, rows)
var = dp_prepare.prepare_variants(
    fg, cfg, view, topo, rows, benefit=100.0, pi_f=0.0, cost_cutoff=None, model=model
)
pack = dp_prepare.prepare_forbidden(frozenset(), fg, rows, topo)

# --- reference sinks
rec = []
real = pricing._Candidate


def spy(rc, delay, label, dl):
    c = real(rc, delay, label, dl)
    rec.append(c)
    return c


ref_labels = {}
real_label = pricing._Label
def label_spy(score, dep, lane, path, paid):
    lab = real_label(score, dep, lane, path, paid)
    key = (dep, -1 if lane is None else lane, tuple((q, r) for q, r in path))
    ref_labels.setdefault(key, score)
    return lab
pricing._Label = label_spy
pricing._Candidate = spy
pricing._best_column(fg, view, 0.0, cfg, 100.0, frozenset(), seed=False,
                     incumbent=None, model=model)
pricing._Candidate = real
pricing._Label = real_label
ref = {
    (c.label.departure_step, c.label.origin_lane_idx, c.dest_lane_idx,
     tuple((q, r) for q, r in c.label.path))
    for c in rec
}

# --- kernel, driving _price_dag directly so the whole label pool is visible
order, bucket_start = dp_kernel._root_buckets(var, topo)
LC, log2cap, CC = 1 << 22, 17, 1 << 20
cap = 1 << log2cap
depth = max(1, topo.state_history_depth)
hop_scratch = max(2, topo.air_hop_limit + 2)
label_score = np.zeros(LC, np.float64)
label_cell = np.zeros(LC, np.int32)
label_parent = np.full(LC, -1, np.int32)
label_hops = np.zeros(LC, np.int32)
label_variant = np.zeros(LC, np.int32)
label_departure = np.zeros(LC, np.int32)
label_lane = np.zeros(LC, np.int32)
label_first_a = np.full(LC, -1, np.int32)
label_first_b = np.full(LC, -1, np.int32)
cand_label = np.zeros(CC, np.int32)
cand_lane = np.zeros(CC, np.int32)
cand_step = np.zeros(CC, np.int32)
out_counts = np.zeros(3, np.int64)
status = dp_kernel._price_dag(
    topo.arc_start, topo.arc_target, topo.arc_roles, topo.hex_remaining,
    topo.dest_mask, topo.dest_lane_start, topo.dest_lane_idx,
    topo.air_hop_limit, topo.revisit_depth, depth, bool(topo.track_first_hop),
    topo.min_step, topo.max_step,
    order, bucket_start,
    var.cell, var.score, var.paid_class, var.departure_step, var.lane_idx,
    var.paid_start, var.paid_cell, var.paid_step, var.paid_value,
    pd.cell_series, pd.series_first, pd.series_start, pd.series_prefix,
    pd.offsets_lo, pd.offsets_hi,
    pack.bits, rows.n_steps, rows.step0,
    float(model.air_weight * cfg.dt_s),
    label_score, label_cell, label_parent, label_hops, label_variant,
    label_departure, label_lane, label_first_a, label_first_b,
    np.full(cap, -1, np.int32), np.zeros(cap, np.uint64),
    np.full(cap, -1, np.int32), np.zeros(cap, np.uint64), log2cap,
    np.zeros(cap, np.int32), np.zeros(cap, np.int32),
    np.zeros(depth, np.int32), np.zeros(depth, np.int32), np.zeros(depth, np.int32),
    np.zeros(hop_scratch, np.int32), np.zeros(hop_scratch, np.int32),
    np.zeros(dp_kernel.FSUM_MAX_PARTIALS, np.float64),
    cand_label, cand_lane, cand_step,
    np.zeros(1, np.uint8), out_counts,
)
n_labels, n_cand = int(out_counts[0]), int(out_counts[1])
print(f"status={status} labels={n_labels} candidates={n_cand}")

cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))


def path_of(label):
    chain = []
    node = label
    while node >= 0:
        chain.append(int(label_cell[node]))
        node = int(label_parent[node])
    return tuple(cells[c] for c in reversed(chain))


kern_sinks = set()
for i in range(n_cand):
    lab = int(cand_label[i])
    kern_sinks.add((
        int(label_departure[lab]),
        None if label_lane[lab] < 0 else int(label_lane[lab]),
        None if cand_lane[i] < 0 else int(cand_lane[i]),
        path_of(lab),
    ))

missing = sorted(ref - kern_sinks)
print(f"reference sinks={len(ref)} kernel sinks={len(kern_sinks)} missing={len(missing)}")
if not missing:
    raise SystemExit(0)

# Every label the kernel built, indexed by (departure, lane, path)
built = {}
for lab in range(n_labels):
    built.setdefault(
        (int(label_departure[lab]), int(label_lane[lab]), path_of(lab)), lab
    )

target = missing[0]
dep, lane, dlane, full = target
print(f"\ntarget: dep={dep} lane={lane} dest_lane={dlane} hops={len(full) - 1}")
print(f"  path {full}")
lane_key = -1 if lane is None else lane
for k in range(1, len(full) + 1):
    prefix = full[:k]
    lab = built.get((dep, lane_key, prefix))
    print(f"  hop {k - 1}: prefix={prefix[-1]} built={'YES lab=%d' % lab if lab is not None else 'NO'}")
    if lab is not None:
        rs = ref_labels.get((dep, lane_key, prefix))
        ks = float(label_score[lab])
        flag = "" if rs is None else ("  SCORES MATCH" if rs == ks else f"  *** SCORE DIFF {ks - rs:+.6e}")
        print(f"        ref_score={rs} kern_score={ks}{flag}")
    if lab is None:
        parent_lab = built.get((dep, lane_key, full[:k - 1]))
        print(f"    -> parent label {parent_lab} exists; the arc to {full[k - 1]} was rejected")
        if parent_lab is not None:
            pc, nc = full[k - 2], full[k - 1]
            pi = cells.index(pc)
            ni = cells.index(nc)
            hops = int(label_hops[parent_lab])
            print(f"    parent hops={hops} step? cell={pc}->{nc} idx {pi}->{ni}")
            arcs = [int(topo.arc_target[a])
                    for a in range(int(topo.arc_start[pi]), int(topo.arc_start[pi + 1]))]
            print(f"    arc present: {ni in arcs}")
            print(f"    hex_remaining[n]={int(topo.hex_remaining[ni])} "
                  f"air_hop_limit={topo.air_hop_limit} "
                  f"hops+1+dtg={hops + 1 + int(topo.hex_remaining[ni])}")
            recent = np.zeros(depth, np.int32)
            nr = dp_kernel._fill_recent(parent_lab, depth, label_parent, label_cell, recent)
            print(f"    recent={[cells[c] for c in recent[:nr].tolist()]} "
                  f"revisit_depth={topo.revisit_depth} "
                  f"banned={ni in recent[:min(nr, topo.revisit_depth)].tolist()}")
        break


# --- who occupies the hop-7 label's dominance slot, and did the reference build it?
tgt = built[(dep, lane_key, full[:8])]
tgt_step = int(var.start_step[int(label_variant[tgt])]) + int(label_hops[tgt])
tgt_key = (int(label_cell[tgt]),
           tuple(np.zeros(depth, np.int32)[:0].tolist()), 0)
rec_buf = np.zeros(depth, np.int32)
n = dp_kernel._fill_recent(tgt, depth, label_parent, label_cell, rec_buf)
tgt_recent = tuple(rec_buf[:n].tolist())
tgt_pc = int(var.paid_class[int(label_variant[tgt])])
tgt_fh = (int(label_first_a[tgt]), int(label_first_b[tgt]))
print(f"\ntarget label {tgt}: step={tgt_step} cell={cells[int(label_cell[tgt])]} "
      f"recent={[cells[c] for c in tgt_recent]} paid_class={tgt_pc} first_hop={tgt_fh}")

rivals = []
for lab in range(n_labels):
    v = int(label_variant[lab])
    if int(var.start_step[v]) + int(label_hops[lab]) != tgt_step:
        continue
    if int(label_cell[lab]) != int(label_cell[tgt]):
        continue
    if int(var.paid_class[v]) != tgt_pc:
        continue
    if (int(label_first_a[lab]), int(label_first_b[lab])) != tgt_fh:
        continue
    m = dp_kernel._fill_recent(lab, depth, label_parent, label_cell, rec_buf)
    if tuple(rec_buf[:m].tolist()) != tgt_recent:
        continue
    rivals.append(lab)

print(f"labels sharing that exact dominance key at that step: {len(rivals)}")
for lab in rivals:
    p = path_of(lab)
    in_ref = (int(label_departure[lab]), int(label_lane[lab]), p) in ref_labels
    print(f"  lab={lab} score={float(label_score[lab]):.9f} hops={int(label_hops[lab])} "
          f"in_reference={in_ref} path={p}")
