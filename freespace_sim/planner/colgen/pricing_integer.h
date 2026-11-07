// Checked two-word arithmetic; see analysis/colgen/GPU_INTEGER_PRUNING.md.
// All word operations are unsigned. Invalid values never authorize a deletion.
struct IntWord { unsigned int low, high; bool valid; };
struct IntInterval { IntWord lower, upper; };

INLINE IntWord iw_invalid() { return {0u, 0u, false}; }
INLINE IntWord iw_make(unsigned int low, unsigned int high) {
    return {low, high, !(low == 0u && high == 0x80000000u)};
}
INLINE bool iw_less(IntWord a, IntWord b) {
    unsigned int ah = a.high ^ 0x80000000u, bh = b.high ^ 0x80000000u;
    return ah < bh || (ah == bh && a.low < b.low);
}
INLINE IntWord iw_add(IntWord a, IntWord b) {
    if (!a.valid || !b.valid) return iw_invalid();
    unsigned int low = a.low + b.low;
    unsigned int high = a.high + b.high + (low < a.low ? 1u : 0u);
    if (((a.high ^ b.high) & 0x80000000u) == 0u &&
        ((a.high ^ high) & 0x80000000u) != 0u) return iw_invalid();
    return iw_make(low, high);
}
INLINE IntWord iw_negate(IntWord a) {
    if (!a.valid) return iw_invalid();
    unsigned int low = ~a.low + 1u;
    return iw_make(low, ~a.high + (low == 0u ? 1u : 0u));
}
INLINE IntWord iw_multiply(IntWord a, unsigned int factor) {
    if (!a.valid) return iw_invalid();
    IntWord result = iw_make(0u, 0u);
    while (factor != 0u) {
        if (factor & 1u) result = iw_add(result, a);
        factor >>= 1;
        // Do not manufacture an overflow by doubling after the last used bit.
        if (factor != 0u) a = iw_add(a, a);
        if (!a.valid || !result.valid) return iw_invalid();
    }
    return result;
}
INLINE IntInterval ii_invalid() { return {iw_invalid(), iw_invalid()}; }
INLINE bool ii_valid(IntInterval a) {
    return a.lower.valid && a.upper.valid && !iw_less(a.upper, a.lower);
}
INLINE IntInterval ii_load(GLOBAL const unsigned int* data) {
    if (data[4] != 1u) return ii_invalid();
    IntInterval value = {iw_make(data[0], data[1]), iw_make(data[2], data[3])};
    return ii_valid(value) ? value : ii_invalid();
}
INLINE void ii_store(GLOBAL unsigned int* data, IntInterval value) {
    bool valid = ii_valid(value);
    data[0] = valid ? value.lower.low : 0u;
    data[1] = valid ? value.lower.high : 0u;
    data[2] = valid ? value.upper.low : 0u;
    data[3] = valid ? value.upper.high : 0u;
    data[4] = valid ? 1u : 0u;
}
INLINE IntInterval ii_add(IntInterval a, IntInterval b) {
    if (!ii_valid(a) || !ii_valid(b)) return ii_invalid();
    return {iw_add(a.lower, b.lower), iw_add(a.upper, b.upper)};
}
INLINE IntInterval ii_subtract(IntInterval a, IntInterval b) {
    IntInterval negative = {iw_negate(b.upper), iw_negate(b.lower)};
    return ii_add(a, negative);
}
INLINE IntInterval ii_multiply(IntInterval a, unsigned int factor) {
    if (!ii_valid(a)) return ii_invalid();
    return {iw_multiply(a.lower, factor), iw_multiply(a.upper, factor)};
}
INLINE unsigned int iw_bits(IntWord a) {
    if (a.high & 0x80000000u) a = iw_negate(a);
    unsigned int bits = a.high == 0u ? 0u : 32u;
    unsigned int word = a.high == 0u ? a.low : a.high;
    while (word != 0u) { ++bits; word >>= 1; }
    return bits;
}
INLINE IntInterval ii_rounded(IntInterval value, int exponent, int direction) {
    if (!ii_valid(value) || exponent < -1074 || exponent > 1023 ||
        direction < -1 || direction > 1) return ii_invalid();
    unsigned int low_bits = iw_bits(value.lower), high_bits = iw_bits(value.upper);
    unsigned int bits = low_bits > high_bits ? low_bits : high_bits;
    if (exponent + (int)bits > 1023) return ii_invalid();
    unsigned int shift = bits > 52u ? bits - 51u : 1u;
    IntWord padding = iw_make(1u << shift, 0u);
    if (direction <= 0) value.lower = iw_add(value.lower, iw_negate(padding));
    if (direction >= 0) value.upper = iw_add(value.upper, padding);
    return value;
}
INLINE bool ii_closes(IntInterval upper, IntInterval cutoff) {
    return ii_valid(upper) && ii_valid(cutoff) && !iw_less(cutoff.lower, upper.upper);
}
