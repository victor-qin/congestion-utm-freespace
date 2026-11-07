// Operands match BoundOperands.packed; absent independent bounds are ignored.
INLINE IntInterval ib_add(IntInterval a, IntInterval b, int exponent, int direction) {
    return ii_rounded(ii_add(a, b), exponent, direction);
}
INLINE IntInterval ib_subtract(IntInterval a, IntInterval b, int exponent, int direction) {
    return ii_rounded(ii_subtract(a, b), exponent, direction);
}
INLINE IntInterval ib_finish(
    IntInterval lower, IntInterval benefit, IntInterval assignment,
    IntInterval allowance, int exponent
) {
    IntInterval upper = ib_subtract(benefit, lower, exponent, 1);
    upper = ib_subtract(upper, assignment, exponent, 1);
    return ib_add(upper, allowance, exponent, 1);
}
INLINE IntInterval ib_affine(
    IntInterval cost, IntInterval prefix, IntInterval suffix, IntInterval negative,
    IntInterval benefit, IntInterval assignment, IntInterval allowance, int exponent
) {
    IntInterval lower = ib_add(cost, prefix, exponent, -1);
    lower = ib_add(lower, suffix, exponent, -1);
    lower = ib_add(lower, negative, exponent, -1);
    return ib_finish(lower, benefit, assignment, allowance, exponent);
}
INLINE IntInterval ib_minimum(IntInterval a, IntInterval b) {
    if (!ii_valid(a)) return b;
    if (!ii_valid(b)) return a;
    return iw_less(b.upper, a.upper) ? b : a;
}
INLINE IntInterval ib_upper(GLOBAL const unsigned int* inputs, int exponent) {
    IntInterval benefit = ii_load(inputs), assignment = ii_load(inputs + 5);
    IntInterval negative = ii_load(inputs + 10), allowance = ii_load(inputs + 15);
    IntInterval cost = ii_load(inputs + 20), affine = ii_load(inputs + 25);
    IntInterval positive = ii_load(inputs + 30), visits = ii_load(inputs + 35);
    IntInterval future = ii_load(inputs + 40), arrival = ii_load(inputs + 45);
    IntInterval arrival_affine = ii_load(inputs + 50);
    IntInterval zero = {iw_make(0u, 0u), iw_make(0u, 0u)};
    IntInterval dual = ib_add(ib_add(zero, positive, exponent, -1), negative, exponent, -1);
    IntInterval upper = ib_subtract(benefit, ii_rounded(cost, exponent, -1), exponent, 1);
    upper = ib_subtract(upper, dual, exponent, 1);
    upper = ib_subtract(upper, assignment, exponent, 1);
    upper = ib_minimum(upper, ib_affine(affine, visits, future, negative,
                                      benefit, assignment, allowance, exponent));
    IntInterval lower = ib_add(cost, positive, exponent, -1);
    lower = ib_add(lower, arrival, exponent, -1);
    lower = ib_add(lower, negative, exponent, -1);
    upper = ib_minimum(upper, ib_finish(lower, benefit, assignment, allowance, exponent));
    return ib_minimum(upper, ib_affine(affine, positive, arrival_affine, negative,
                                     benefit, assignment, allowance, exponent));
}
