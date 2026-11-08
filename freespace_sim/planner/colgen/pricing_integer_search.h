// Exact membership and conservative arithmetic for the opt-in integer search.
INLINE IntInterval is_clamp_zero(IntInterval value) {
    if (!ii_valid(value)) return ii_invalid();
    IntWord zero = iw_make(0u, 0u);
    if (iw_less(value.lower, zero)) value.lower = zero;
    if (iw_less(value.upper, zero)) value.upper = zero;
    return value;
}
INLINE IntInterval is_sum_lower(IntInterval sum, unsigned int count, int exponent) {
    if (!ii_valid(sum)) return ii_invalid();
    IntInterval padded = ii_rounded(sum, exponent, -1);
    if (!ii_valid(padded)) return ii_invalid();
    IntWord error = iw_add(padded.lower, iw_negate(sum.lower));
    sum.lower = iw_add(sum.lower, iw_multiply(error, count));
    return is_clamp_zero(sum);
}
INLINE IntInterval is_range_min(
    GLOBAL const unsigned int* tree, int leaves, int first, int last
) {
    first += leaves; last += leaves + 1;
    IntInterval result = ii_invalid();
    bool found = false;
    while (first < last) {
        for (int side = 0; side < 2; ++side) {
            int index = -1;
            if (side == 0 && (first & 1)) index = first++;
            if (side == 1 && (last & 1)) index = --last;
            if (index < 0) continue;
            IntInterval value = ii_load(tree + index * 5);
            // Invalid numeric data is not interchangeable with a proved +infinity leaf.
            if (!ii_valid(value)) return ii_invalid();
            if (!found) { result = value; found = true; }
            else {
                if (iw_less(value.lower, result.lower)) result.lower = value.lower;
                if (iw_less(value.upper, result.upper)) result.upper = value.upper;
            }
        }
        first /= 2; last /= 2;
    }
    return result;
}
INLINE void integer_expand_one(
    unsigned int tid, GLOBAL const int* control, GLOBAL const int* labels,
    GLOBAL const float* values, GLOBAL const int* graph, GLOBAL const int* roots,
    GLOBAL const float* root_values, GLOBAL const float* visits, GLOBAL const int* blocked,
    GLOBAL const int* paid_rows, GLOBAL const float* paid_values,
    GLOBAL int* output, GLOBAL float* scores,
    GLOBAL const int* price_map, GLOBAL const int* price_ranges,
    GLOBAL const int* price_steps, GLOBAL const float* price_values, GLOBAL const float* cutoffs,
    GLOBAL const unsigned int* visit_masks, GLOBAL const int* endpoint_meta,
    GLOBAL const int* endpoint_entries, GLOBAL const float* endpoint_trees,
    GLOBAL const int* numeric_meta, GLOBAL const unsigned int* numeric_roots,
    GLOBAL const unsigned int* numeric_costs, GLOBAL const int* numeric_prices,
    GLOBAL const unsigned int* numeric_price_values,
    GLOBAL const unsigned int* numeric_endpoints, GLOBAL const unsigned int* numeric_future,
    GLOBAL const unsigned int* numeric_cutoffs
) {
    if (tid >= (unsigned int)(control[0] * 6)) return;
    int stride = control[1], source = ((int)tid / 6) * stride, target = (int)tid * stride;
    int flags = output[target + 6];
    if (!flags) return;
    // FP32 supplies ranking only; discard every FP32 numerical decision flag.
    flags &= 3;
    int cell = output[target], root = output[target + 1], hops = output[target + 2];
    int step = control[2] + 1, remaining = graph[cell * 14 + 12];
    GLOBAL const int* meta = numeric_meta + root * 4;
    int exponent = meta[0], limit = meta[1], cost_offset = meta[2];
    int numeric_offset = stride - 15;
    IntInterval sum = {iw_make(0u, 0u), iw_make(0u, 0u)}, union_sum = sum;
    unsigned int count = 0u, union_count = 0u;
    int first = numeric_prices[1 + cell * 2], last = numeric_prices[2 + cell * 2];
    int times = 1 + numeric_prices[0] * 2;
    int stop = last;
    while (first < last) {
        int mid = first + (last - first) / 2;
        if (numeric_prices[times + mid] < step + control[3]) first = mid + 1;
        else last = mid;
    }
    for (int index = first; index < stop; ++index) {
        int time = numeric_prices[times + index];
        if (time > step + control[4]) break;
        IntInterval price = ii_load(numeric_price_values + index * 5);
        sum = ii_add(sum, price); ++count;
        bool paid = false;
        for (int item = roots[root * 6 + 1]; item < roots[root * 6 + 2]; ++item) {
            if (paid_rows[item * 2] == cell && paid_rows[item * 2 + 1] == time) {
                paid = true; break;
            }
        }
        if (!paid) { union_sum = ii_add(union_sum, price); ++union_count; }
    }
    IntInterval positive = is_sum_lower(union_sum, union_count, exponent);
    IntInterval visit = is_sum_lower(sum, count, exponent);
    GLOBAL const unsigned int* previous = (GLOBAL const unsigned int*)(labels + source + numeric_offset);
    positive = is_clamp_zero(ib_add(ii_load(previous), positive, exponent, -1));
    visit = is_clamp_zero(ib_add(ii_load(previous + 5), visit, exponent, -1));
    GLOBAL unsigned int* numeric = (GLOBAL unsigned int*)(output + target + numeric_offset);
    ii_store(numeric, positive); ii_store(numeric + 5, visit);
    IntInterval upper = ii_invalid();
    if (cost_offset >= 0 && hops + remaining <= limit) {
        GLOBAL const unsigned int* constants = numeric_roots + root * 35;
        IntInterval benefit = ii_load(constants), assignment = ii_load(constants + 5);
        IntInterval negative = ii_load(constants + 10), allowance = ii_load(constants + 15);
        IntInterval slope = ii_load(constants + 20);
        IntInterval cost = ii_load(numeric_costs + (cost_offset + 2 * (hops + remaining)) * 5);
        IntInterval affine = ii_load(numeric_costs + (cost_offset + 2 * hops + 1) * 5);
        IntInterval q = ii_invalid(), d = ii_invalid();
        int flight = roots[root * 6];
        GLOBAL const int* endpoint = endpoint_meta + flight * 5;
        int arrival_last = roots[root * 6 + 3], available = limit - hops;
        if (step <= 2147483647 - available) arrival_last = min(arrival_last, step + available);
        if (endpoint[0] <= step && step + remaining <= arrival_last && arrival_last <= endpoint[1]) {
            for (int entry = endpoint[3]; entry < endpoint[4]; ++entry) {
                GLOBAL const int* item = endpoint_entries + entry * 4;
                if (!item[3] || item[0] != output[target + 7] || item[1] != output[target + 8]) continue;
                GLOBAL const unsigned int* tree = numeric_endpoints + item[2] * 5;
                q = is_range_min(tree, endpoint[2], step + remaining - endpoint[0], arrival_last - endpoint[0]);
                IntInterval h = is_range_min(tree + 2 * endpoint[2] * 5, endpoint[2],
                                             step + remaining - endpoint[0], arrival_last - endpoint[0]);
                IntInterval elapsed = ii_rounded(ii_multiply(slope, (unsigned int)(step - endpoint[0])), exponent, 1);
                d = is_clamp_zero(ib_subtract(h, elapsed, exponent, -1));
                break;
            }
        }
        upper = ib_evaluate(benefit, assignment, negative, allowance, cost, affine,
                            positive, visit, ii_load(numeric_future + cell * 5), q, d, exponent);
        if (ii_closes(upper, ii_load(numeric_cutoffs + flight * 5))) flags |= meta[3] ? (4 | 128) : 64;
    }
    ii_store(numeric + 10, upper);
    output[target + 6] = flags;
}
