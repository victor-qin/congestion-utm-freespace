// Shared Metal/CUDA body. Address spaces, entry points and atomics are runtime adapters.
// Label fields: cell, root, hops, stable ID/parent ID, first A/B, flags, recent cells.
// Root values: benefit, assignment, ground, negative credit, hop cost, cutoff,
// affine air-cost offset/slope, positive origin price, minimum total hops, future cost.
INLINE float completion_bound(GLOBAL const float* data, float hops, float positive) {
    float air = fmax(0.0f, data[6] + hops * data[7]);
    return ((data[0] - (data[2] + air)) - (positive - data[3])) - data[1];
}

INLINE void root_bound_one(unsigned int tid, GLOBAL const int* control,
                          GLOBAL const float* roots, GLOBAL float* bounds) {
    if (tid >= (unsigned int)control[0]) return;
    GLOBAL const float* data = roots + tid * 11;
    float upper = completion_bound(data, data[9], data[8]);
    // Infinity also marks an unavailable table; it must never authorize deletion.
    if (isfinite(data[10])) {
        float future = ((data[0] - data[2]) - (data[6] + data[8] + data[10] - data[3])) - data[1];
        upper = fmin(upper, future);
    }
    bounds[tid] = upper;
}

INLINE float lower_sum(float a, float b) {
    if (isinf(a) || isinf(b)) return a + b;
    float sum = a + b;
    return sum > 0.0f ? NEXT_DOWN(sum) : 0.0f;
}

INLINE void future_one(unsigned int tid, GLOBAL const int* control,
                       GLOBAL const int* graph, GLOBAL const float* next,
                       GLOBAL const float* visits, GLOBAL const float* air,
                       GLOBAL float* output) {
    if (tid >= (unsigned int)control[0]) return;
    float best = graph[tid * 7 + 6] ? air[1] : INFINITY;
    for (int arc = 0; arc < 6; ++arc) {
        int target = graph[tid * 7 + arc];
        if (target >= 0) best = fmin(best, lower_sum(lower_sum(air[0], visits[target]), next[target]));
    }
    output[tid] = best;
}

INLINE void expand_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL const int* labels,
    GLOBAL const float* values, GLOBAL const int* graph, GLOBAL const int* roots,
    GLOBAL const float* root_values, GLOBAL const float* visits, GLOBAL const int* blocked,
    GLOBAL const int* paid_rows, GLOBAL const float* paid_values,
    GLOBAL int* output, GLOBAL float* scores
) {
    int n = control[0], stride = control[1], step = control[2];
    if (tid >= (unsigned int)(n * 6)) return;
    int src_index = (int)tid / 6, arc = (int)tid % 6;
    int a = src_index * stride, b = (int)tid * stride;
    output[b + 6] = 0;
    int cell = labels[a], root = labels[a + 1], hops = labels[a + 2];
    int target = graph[cell * 14 + arc];
    if (target < 0) return;
    int role = graph[cell * 14 + 6 + arc];
    int remaining = graph[target * 14 + 12];
    if (step + 1 + remaining > roots[root * 6 + 3] ||
        hops + 1 + remaining > roots[root * 6 + 4] || blocked[target]) return;
    int depth = roots[root * 6 + 5];
    for (int j = 0; j < depth; ++j) if (labels[a + 7 + j] == target) return;
    bool finish = graph[target * 14 + 13] && (role & (hops == 0 ? 8 : 4));
    bool onward = (role & (hops == 0 ? 2 : 1)) != 0;
    if (!finish && !onward) return;
    float charge = visits[target * 3];
    for (int j = roots[root * 6 + 1]; j < roots[root * 6 + 2]; ++j) {
        if (paid_rows[j * 2] == target && paid_rows[j * 2 + 1] >= step + 1 + control[3]
            && paid_rows[j * 2 + 1] <= step + 1 + control[4]) charge -= paid_values[j];
    }
    GLOBAL const float* data = root_values + root * 11;
    float score = (values[src_index * 2] - data[4]) - charge;
    float positive = values[src_index * 2 + 1] + visits[target * 3 + 1];
    float upper = completion_bound(data,
                                   (float)(hops + 1 + remaining), positive);
    if (isfinite(visits[target * 3 + 2])) {
        float future = ((data[0] - data[2]) -
                        (data[6] + (float)(hops + 1) * data[7] + positive + visits[target * 3 + 2] - data[3])) - data[1];
        upper = fmin(upper, future);
    }
    int flags = (finish ? 1 : 0) | (onward ? 2 : 0);
    if (upper <= data[5]) flags |= 4;
    output[b] = target;
    output[b + 1] = root;
    output[b + 2] = hops + 1;
    output[b + 3] = labels[a + 3];
    output[b + 4] = hops == 0 ? cell : labels[a + 4];
    output[b + 5] = hops == 0 ? target : labels[a + 5];
    output[b + 6] = flags;
    output[b + 7] = target;
    for (int j = 8; j < stride; ++j) output[b + j] = labels[a + j - 1];
    scores[tid * 3] = score;
    scores[tid * 3 + 1] = positive;
    scores[tid * 3 + 2] = upper;
}

INLINE bool same_state(GLOBAL const int* labels, int a, int b, int stride) {
    // Root identity refines the production key, keeping different departures/lanes separate.
    for (int j = 0; j < stride; ++j) {
        if (j == 3 || j == 6) continue;
        if (labels[a * stride + j] != labels[b * stride + j]) return false;
    }
    return true;
}

INLINE void group_one(unsigned int tid, GLOBAL const int* control,
                      GLOBAL const int* labels, GLOBAL const float* scores,
                      GLOBAL int* table) {
    int n = control[0], stride = control[1], capacity = control[2];
    if (tid >= (unsigned int)n || !(labels[tid * stride + 6] & 2)) return;
    unsigned int hash = 2166136261u;
    for (int j = 0; j < stride; ++j) {
        if (j == 3 || j == 6) continue;
        hash = (hash ^ (unsigned int)labels[tid * stride + j]) * 16777619u;
    }
    int slot = (int)(hash & (unsigned int)(capacity - 1));
    for (int probe = 0; probe < capacity; ++probe) {
        int old = compare_exchange(table + slot, -1, (int)tid);
        if (old == -1 || old == (int)tid) return;
        if (same_state(labels, old, (int)tid, stride)) {
            while (true) {
                float previous = scores[old * 3], candidate = scores[tid * 3];
                if (!(candidate > previous || (candidate == previous && (int)tid < old))) return;
                int observed = compare_exchange(table + slot, old, (int)tid);
                if (observed == old) return;
                old = observed;
            }
        }
        slot = (slot + 1) & (capacity - 1);
    }
}
