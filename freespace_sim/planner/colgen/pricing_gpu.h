// Shared Metal/CUDA body. Address spaces, entry points and atomics are runtime adapters.
// Label fields: cell, root, hops, stable ID/parent ID, first A/B, status, paid low/high, recent cells.
// Root values: benefit, assignment, ground, negative credit, hop cost, cutoff,
// affine air-cost offset/slope, positive origin price, minimum total hops, future cost.
INLINE float completion_bound(GLOBAL const float* data, float hops, float positive) {
    float air = fmax(0.0f, data[6] + hops * data[7]);
    return ((data[0] - (data[2] + air)) - (positive - data[3])) - data[1];
}

INLINE float range_min(GLOBAL const float* tree, int leaves, int first, int last) {
    first += leaves;
    last += leaves + 1;
    float value = INFINITY;
    while (first < last) {
        if (first & 1) value = fmin(value, tree[first++]);
        if (last & 1) value = fmin(value, tree[--last]);
        first /= 2;
        last /= 2;
    }
    return value;
}

INLINE float endpoint_bound(
    GLOBAL const float* data, GLOBAL const int* root, int step, int hops, int remaining,
    float positive, unsigned int low, unsigned int high, GLOBAL const int* meta,
    GLOBAL const int* entries, GLOBAL const float* trees
) {
    GLOBAL const int* flight = meta + root[0] * 5;
    int first = step + remaining, last = root[3], available = root[4] - hops;
    if (available < 0) return -INFINITY;
    // The hop-limit date can exceed int32 even when the earlier horizon is representable.
    if (step <= 2147483647 - available) last = min(last, step + available);
    if (first > last) return -INFINITY;
    if (step < flight[0] || last > flight[1]) return INFINITY;
    for (int entry = flight[3]; entry < flight[4]; ++entry) {
        GLOBAL const int* item = entries + entry * 4;
        if (!item[3] || (unsigned int)item[0] != low || (unsigned int)item[1] != high) continue;
        GLOBAL const float* tree = trees + item[2];
        float q = range_min(tree, flight[2], first - flight[0], last - flight[0]);
        float h = range_min(tree + 2 * flight[2], flight[2], first - flight[0], last - flight[0]);
        float d = fmax(0.0f, h - (float)(step - flight[0]) * data[7]);
        float clamped = completion_bound(data, (float)(hops + remaining), positive + q);
        float affine = ((data[0] - data[2]) -
                        (data[6] + (float)hops * data[7] + positive + d - data[3])) - data[1];
        return fmin(clamped, affine);
    }
    // A cache miss is weaker information, never a proof of infeasibility.
    return INFINITY;
}

INLINE void root_bound_one(unsigned int tid, GLOBAL const int* control,
                          GLOBAL const float* roots, GLOBAL float* bounds,
                          GLOBAL const int* root_index, GLOBAL const int* root_endpoint,
                          GLOBAL const int* meta, GLOBAL const int* entries,
                          GLOBAL const float* trees) {
    if (tid >= (unsigned int)control[0]) return;
    GLOBAL const float* data = roots + tid * 11;
    float upper = completion_bound(data, data[9], data[8]);
    // Infinity also marks an unavailable table; it must never authorize deletion.
    if (isfinite(data[10])) {
        float future = ((data[0] - data[2]) - (data[6] + data[8] + data[10] - data[3])) - data[1];
        upper = fmin(upper, future);
    }
    upper = fmin(upper, endpoint_bound(
        data, root_index + tid * 6, root_endpoint[tid * 3 + 2], 0, (int)data[9], data[8],
        (unsigned int)root_endpoint[tid * 3], (unsigned int)root_endpoint[tid * 3 + 1],
        meta, entries, trees));
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
    GLOBAL int* output, GLOBAL float* scores,
    GLOBAL const int* price_map, GLOBAL const int* price_ranges,
    GLOBAL const int* price_steps, GLOBAL const float* price_values,
    GLOBAL const unsigned int* visit_masks,
    GLOBAL const int* endpoint_meta, GLOBAL const int* endpoint_entries,
    GLOBAL const float* endpoint_trees
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
    for (int j = 0; j < depth; ++j) if (labels[a + 9 + j] == target) return;
    bool finish = graph[target * 14 + 13] && (role & (hops == 0 ? 8 : 4));
    bool onward = (role & (hops == 0 ? 2 : 1)) != 0;
    if (!finish && !onward) return;
    int slot = price_map[target];
    int lo = price_ranges[slot * 2], stop = price_ranges[slot * 2 + 1], hi = stop;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (price_steps[mid] < step + 1) lo = mid + 1;
        else hi = mid;
    }
    bool priced = lo < stop && price_steps[lo] == step + 1;
    float charge = priced ? price_values[lo * 2] : 0.0f;
    float positive_charge = priced ? price_values[lo * 2 + 1] : 0.0f;
    for (int j = roots[root * 6 + 1]; j < roots[root * 6 + 2]; ++j) {
        if (paid_rows[j * 2] == target && paid_rows[j * 2 + 1] >= step + 1 + control[3]
            && paid_rows[j * 2 + 1] <= step + 1 + control[4]) {
            charge -= paid_values[j];
            positive_charge -= fmax(0.0f, paid_values[j]);
        }
    }
    GLOBAL const float* data = root_values + root * 11;
    float score = (values[src_index * 2] - data[4]) - charge;
    float positive = values[src_index * 2 + 1] + fmax(0.0f, positive_charge);
    float upper = completion_bound(data,
                                   (float)(hops + 1 + remaining), positive);
    if (isfinite(visits[target])) {
        float future = ((data[0] - data[2]) -
                        (data[6] + (float)(hops + 1) * data[7] + positive + visits[target] - data[3])) - data[1];
        upper = fmin(upper, future);
    }
    int flags = (finish ? 1 : 0) | (onward ? 2 : 0);
    unsigned int low = (unsigned int)labels[a + 7] | visit_masks[target * 2];
    unsigned int high = (unsigned int)labels[a + 8] | visit_masks[target * 2 + 1];
    float endpoint = endpoint_bound(data, roots + root * 6, step + 1, hops + 1,
                                   remaining, positive, low, high,
                                   endpoint_meta, endpoint_entries, endpoint_trees);
    if (endpoint <= data[5] && upper > data[5]) flags |= 8;
    upper = fmin(upper, endpoint);
    if (upper <= data[5]) flags |= 4;
    output[b] = target;
    output[b + 1] = root;
    output[b + 2] = hops + 1;
    output[b + 3] = labels[a + 3];
    output[b + 4] = hops == 0 ? cell : labels[a + 4];
    output[b + 5] = hops == 0 ? target : labels[a + 5];
    output[b + 6] = flags;
    output[b + 7] = (int)low;
    output[b + 8] = (int)high;
    output[b + 9] = target;
    for (int j = 10; j < stride; ++j) output[b + j] = labels[a + j - 1];
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
