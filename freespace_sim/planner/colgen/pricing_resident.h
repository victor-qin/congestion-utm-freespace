// Record width is supplied by the caller; extra endpoint/mask words are opaque here.
INLINE void resident_load_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL const int* arena,
    GLOBAL const float* arena_values, GLOBAL const int* frontier, GLOBAL const int* counts,
    GLOBAL int* labels, GLOBAL float* values
) {
    int n = control[0], stride = control[1], start = control[2];
    if (tid >= (unsigned int)n) return;
    int offset = (int)tid * stride;
    labels[offset + 6] = 0;
    if (counts[4] || start + (int)tid >= counts[1]) return;
    int id = frontier[start + (int)tid];
    for (int j = 0; j < stride; ++j) labels[offset + j] = arena[id * stride + j];
    labels[offset + 3] = id;
    values[tid * 2] = arena_values[id * 2];
    values[tid * 2 + 1] = arena_values[id * 2 + 1];
}

INLINE void resident_compact_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL const int* records,
    GLOBAL const float* scores, GLOBAL int* arena, GLOBAL float* arena_values,
    GLOBAL int* next, GLOBAL int* counts, GLOBAL int* event_records,
    GLOBAL float* event_values, GLOBAL int* event_meta, GLOBAL float* event_upper
) {
    int n = control[0], stride = control[1], capacity = control[2], queue_capacity = control[3];
    if (tid >= (unsigned int)n) return;
    int source = (int)tid * stride, flags = records[source + 6];
    if (!flags) return;
    fetch_add(counts + 5, 1);
    if (flags & 16) {
        fetch_add(counts + 6, 1);
        return;
    }
    // Numerical proposals remain pending; their full state survives until verification.
    if (flags & 5) {
        int event = fetch_add(counts + 3, 1);
        if (event >= queue_capacity) {
            compare_exchange(counts + 4, 0, 2);
            return;
        }
        for (int j = 0; j < stride; ++j) event_records[event * stride + j] = records[source + j];
        event_values[event * 2] = scores[tid * 3];
        event_values[event * 2 + 1] = scores[tid * 3 + 1];
        event_upper[event] = scores[tid * 3 + 2];
        for (int j = 0; j < 4; ++j) event_meta[event * 5 + j] = records[source + j];
        event_meta[event * 5 + 4] = flags;
    }
    if (!(flags & 4) && (flags & 2)) {
        int id = fetch_add(counts, 1);
        int slot = fetch_add(counts + 2, 1);
        if (id >= capacity || slot >= capacity) {
            compare_exchange(counts + 4, 0, 1);
            return;
        }
        for (int j = 0; j < stride; ++j) arena[id * stride + j] = records[source + j];
        arena_values[id * 2] = scores[tid * 3];
        arena_values[id * 2 + 1] = scores[tid * 3 + 1];
        next[slot] = id;
    }
}

INLINE void resident_copy_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL const int* records,
    GLOBAL const float* values, GLOBAL const int* indices, GLOBAL int* arena,
    GLOBAL float* arena_values, GLOBAL int* frontier
) {
    int n = control[0], stride = control[1];
    if (tid >= (unsigned int)n) return;
    int source = control[4] ? indices[tid] : (int)tid;
    int id = control[2] + (int)tid;
    for (int j = 0; j < stride; ++j) arena[id * stride + j] = records[source * stride + j];
    arena_values[id * 2] = values[source * 2];
    arena_values[id * 2 + 1] = values[source * 2 + 1];
    frontier[control[3] + (int)tid] = id;
}

INLINE void resident_advance_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL int* counts
) {
    if (tid) return;
    counts[1] = counts[2];
    counts[2] = control[0];
}
