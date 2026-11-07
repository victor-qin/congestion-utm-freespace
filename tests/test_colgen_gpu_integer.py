"""Exact-oracle containment, failure behavior and actual Metal checks for integer intervals."""

from fractions import Fraction
from dataclasses import replace
import itertools
import math
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import pytest

from freespace_sim.planner.colgen import gpu_runtime
from freespace_sim.planner.colgen.gpu_integer import IntegerInterval, IntegerScale, LIMIT, WORD_MASK
from freespace_sim.planner.colgen.gpu_integer_bounds import BoundOperands, reference_upper


def _contains(scale: IntegerScale, interval: IntegerInterval, exact: Fraction) -> None:
    """Require a valid interval to contain an independent exact value."""
    assert interval.valid
    assert interval.lower * scale.quantum <= exact <= interval.upper * scale.quantum


@pytest.mark.parametrize("exponent", [-1074, -1022, -100, -40, 0, 970, 1023])
def test_conversion_contains_original_binary64_and_roundtrips_words(exponent: int) -> None:
    """Floor/ceiling enclose original values without a rounded scaling operation."""
    scale = IntegerScale(exponent)
    rng = random.Random(113)
    values = [0.0, -0.0, math.ulp(0.0), -math.ulp(0.0), 1.0, -1.0,
              math.nextafter(1.0, 0.0), math.nextafter(1.0, math.inf), 1e-300, 1e300]
    values += [math.ldexp(rng.uniform(-1.0, 1.0), rng.randrange(-1073, 1024))
               for _ in range(1000)]
    packed = scale.encode_array(np.asarray(values))
    for index, value in enumerate(values):
        interval = scale.encode(value)
        np.testing.assert_array_equal(packed[index], interval.packed())
        exact = Fraction.from_float(value) / scale.quantum
        fits = -LIMIT <= math.floor(exact) <= math.ceil(exact) <= LIMIT
        assert interval.valid == fits
        assert IntegerInterval.unpacked(interval.packed()) == interval
        if fits:
            _contains(scale, interval, Fraction.from_float(value))
            assert interval.upper - interval.lower <= 1
            assert scale.upper_float(interval) >= value
    for value in (math.nan, math.inf, -math.inf):
        assert not scale.encode(value).valid


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_arithmetic_encloses_ordered_fp64_and_exact_oracle(direction: int) -> None:
    """Directed operations cover ties, cancellation, subnormals and extreme magnitudes."""
    rng = random.Random(919)
    for exponent in (-1074, -1022, -400, -40, 0, 900, 960):
        scale = IntegerScale(exponent)
        for _ in range(700):
            units = [rng.randrange(-2**61, 2**61), rng.randrange(-2**61, 2**61)]
            first, second = [float(u * scale.quantum) for u in units]
            if rng.random() < 0.3:
                second = -first
            a, b = scale.encode(first), scale.encode(second)
            factor = rng.choice((0, 1, 2, 3, 31, WORD_MASK))
            cases = (
                (scale.add(a, b, direction=direction), first + second,
                 Fraction(first) + Fraction(second)),
                (scale.subtract(a, b, direction=direction), first - second,
                 Fraction(first) - Fraction(second)),
                (scale.multiply(a, factor, direction=direction), first * factor,
                 Fraction(first) * factor),
            )
            for interval, reference, exact in cases:
                if interval.valid:
                    if direction:
                        reference = math.nextafter(reference, math.inf * direction)
                    assert math.isfinite(reference)
                    _contains(scale, interval, Fraction(reference))
                    # The directed interval also encloses the exact operation.
                    _contains(scale, interval, exact)


def test_checked_limits_and_conservative_cutoff_cannot_close_ambiguity() -> None:
    """Overflow, nonfinite cutoffs and uncertain ties remain live instead of saturating."""
    scale = IntegerScale(-40)
    maximum = IntegerInterval.checked(LIMIT, LIMIT)
    minimum = IntegerInterval.checked(-LIMIT, -LIMIT)
    one = IntegerInterval.checked(1, 1)
    assert not scale.add(maximum, one).valid
    assert not scale.subtract(minimum, one).valid
    assert not scale.multiply(maximum, 2).valid
    assert not scale.rounded(maximum, 1).valid
    assert not scale.multiply(IntegerInterval(), 0).valid
    assert not IntegerInterval.checked(-2**63, -2**63).valid
    assert not IntegerScale(1023).rounded(one, 1).valid
    assert scale.upper_float(IntegerInterval()) == math.inf
    cutoff = 3.0 + 1e-6
    encoded = scale.encode(cutoff)
    assert scale.closes(IntegerInterval.checked(encoded.lower, encoded.lower), cutoff)
    assert not scale.closes(IntegerInterval.checked(encoded.lower, encoded.upper + 1), cutoff)
    for invalid in (IntegerInterval(), maximum):
        assert not scale.closes(invalid, cutoff)
    for value in (math.nan, math.inf, -math.inf):
        assert not scale.closes(one, value)


def test_snapshot_scale_has_exact_accumulation_headroom_and_checks_bad_estimates() -> None:
    """Price magnitude/sign changes select fresh units; a bad size estimate still cannot wrap."""
    for values in ((0.0,), (math.ulp(0.0),), (1.0, -1e-200), (1e300, -1e300)):
        for terms in (1, 33, 100000):
            scale = IntegerScale.for_values(values, max_terms=terms)
            for value in values:
                encoded = scale.encode(value)
                assert scale.multiply(encoded, terms).valid
                _contains(scale, encoded, Fraction(value))
    scale = IntegerScale.for_values([1.0])
    assert not scale.encode(1e300).valid
    assert IntegerScale.for_values([1e300]) != scale
    with pytest.raises(ValueError):
        IntegerScale.for_values([math.inf])
    with pytest.raises(ValueError):
        IntegerScale.for_values([1.0], max_terms=0)


@pytest.mark.parametrize("weight", [math.ulp(0.0), 1e-100, 1., 1e100, 1e300])
def test_prepared_hop_costs_preserve_original_fp64_semantics(weight: float) -> None:
    """Vector preparation must preserve original geometry allowances at tiny/extreme weights."""
    from freespace_sim.planner.colgen.gpu_integer_bounds import cost_operands
    from freespace_sim.planner.colgen.network import build_flight_graph
    from tests.test_colgen_gpu_bounds import _bounds
    from tests.test_colgen_pricing_platform import _problem

    problem = _problem()
    cfg = replace(problem.cfg, cost_ground_delay_per_s=weight,
                  cost_air_hold_per_s=weight, cost_air_lateral_per_s=weight)
    graph = build_flight_graph(problem.graph.request, cfg, (), problem.params)
    bounds, variants = _bounds(replace(problem, graph=graph, cfg=cfg))
    for missing in (False, True):
        if missing:
            bounds._length_offset = np.full(variants.n_variants, -math.inf)
        for variant in range(variants.n_variants):
            values = cost_operands(bounds, variant)
            for hops, (cost, affine) in enumerate(values):
                assert cost == bounds.cost_lower(variant, hops)
                first = math.nextafter(float(variants.ground_delay_s[variant])
                                       + float(bounds.air_offset[variant]), -math.inf)
                assert affine == math.nextafter(
                    first + math.nextafter(hops * bounds.air_per_hop, -math.inf), -math.inf,
                )


@pytest.mark.parametrize("magnitude", [math.ulp(0.0), 1e-300, 1.0, 1e8, 1e280])
def test_positive_sum_covers_all_reference_orders(magnitude: float) -> None:
    """A different exact row iteration order cannot tighten the reference prefix unsafely."""
    values = (0.0, magnitude, magnitude / 3.0, magnitude * 2.0, magnitude * 2.0**-53)
    scale = IntegerScale.for_values(values, max_terms=len(values))
    interval = scale.positive_sum(values)
    for order in itertools.permutations(values):
        reference = 0.0
        for value in order:
            if value > 0:
                reference = math.nextafter(reference + value, -math.inf)
        _contains(scale, interval, Fraction(max(0.0, reference)))
    assert not scale.positive_sum((magnitude, -magnitude)).valid
    assert not scale.positive_sum((math.inf,)).valid


def _bound_probe_source(backend: str) -> str:
    """Replace only the development entry point with the complete bound-expression probe."""
    source = _probe_source(backend).replace("void bound_roots(", "void unused_interval_probe(")
    source += Path(gpu_runtime.__file__).with_name("pricing_integer_bounds.h").read_text()
    arguments = ("const int* exponents", "const unsigned int* operands", "unsigned int* output")
    if backend == "metal":
        args = ", ".join(f"device {arg} [[buffer({i})]]" for i, arg in enumerate(arguments))
        source += f"\nkernel void bound_roots({args}, uint tid [[thread_position_in_grid]]) {{\n"
    else:
        source += ('\nextern "C" __global__ void bound_roots(' + ", ".join(arguments)
                   + ") {\nunsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
    return source + "ii_store(output + tid * 5, ib_upper(operands + tid * 55, exponents[tid]));\n}\n"


def test_metal_bound_expressions_enclose_original_fp64_operations(monkeypatch) -> None:
    """Independent original FP64 expressions cover cancellation, missing bounds and overflow."""
    def down(value: float) -> float:
        """Apply the reference's original downward binary64 operation."""
        return math.nextafter(value, -math.inf)

    def up(value: float) -> float:
        """Apply the reference's original upward binary64 operation."""
        return math.nextafter(value, math.inf)

    monkeypatch.setattr(gpu_runtime, "_source", _bound_probe_source)
    runtime = gpu_runtime.ComputeRuntime("metal")
    rng = random.Random(54)
    packed, exponents, expected, references = [], [], [], []
    for magnitude in (math.ulp(0.0), 1e-300, 1.0, 1e8, 1e280):
        for sample in range(100):
            values = [rng.uniform(0.0, 1.0) * magnitude for _ in range(11)]
            values[2] = -values[2]
            values[5] = -values[5]
            values[1] = values[0] if sample % 2 else values[1]
            scale = IntegerScale.for_values(values, max_terms=32)
            if sample % 7 == 0:
                values[8:] = [math.inf] * 3
            intervals = [scale.encode(value) for value in values]
            operands = BoundOperands(*intervals)
            expected.append(reference_upper(scale, operands).packed())
            exponents.append(scale.exponent)
            packed.append(operands.packed())
            benefit, pi, negative, allowance, cost, affine, positive, visits, future, q, d = values
            basic = up(up(up(benefit - down(cost)) - down(down(positive) + negative)) - pi)
            candidates = [basic]
            for prefix, suffix in ((visits, future), (positive, d)):
                if math.isfinite(suffix):
                    lower = down(down(down(affine + prefix) + suffix) + negative)
                    candidates.append(up(up(up(benefit - lower) - pi) + allowance))
            if math.isfinite(q):
                lower = down(down(down(cost + positive) + q) + negative)
                candidates.append(up(up(up(benefit - lower) - pi) + allowance))
            references.append(min(candidates))
    (actual,) = runtime.run("bound_roots", len(packed), (
        np.asarray(exponents, np.int32), np.asarray(packed, np.uint32),
        np.empty((len(packed), 5), np.uint32),
    ), (2,))
    np.testing.assert_array_equal(actual, expected)
    for exponent, words, reference in zip(exponents, actual, references, strict=True):
        interval = IntegerInterval.unpacked(words)
        assert interval.valid
        assert interval.upper * IntegerScale(exponent).quantum >= Fraction(reference)


def _probe_source(backend: str) -> str:
    """Compile prototype arithmetic into a development-only entry point.

    Parameters
    ------------
    - backend (str): source adapter; production search kernels remain unmodified on disk.

    Return
    --------
    - output (str): ordinary runtime source with bound_roots replaced only in this test module.
    """
    source = _ORIGINAL_SOURCE(backend).replace("void bound_roots(", "void unused_bound_roots(")
    source += Path(gpu_runtime.__file__).with_name("pricing_integer.h").read_text()
    arguments = (
        "const unsigned int* control", "const unsigned int* first", "const unsigned int* second",
        "unsigned int* output", "unsigned int* closed",
    )
    if backend == "metal":
        args = ", ".join(f"device {arg} [[buffer({i})]]" for i, arg in enumerate(arguments))
        source += f"\nkernel void bound_roots({args}, uint tid [[thread_position_in_grid]]) {{\n"
    else:
        source += ('\nextern "C" __global__ void bound_roots(' + ", ".join(arguments)
                   + ") {\nunsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
    return source + """
    IntInterval a = ii_load(first + tid * 5), b = ii_load(second + tid * 5);
    unsigned int op = control[tid * 4], factor = control[tid * 4 + 2];
    int exponent = (int)control[tid * 4 + 1], direction = (int)control[tid * 4 + 3];
    IntInterval result = op == 0u ? ii_add(a, b) :
                         op == 1u ? ii_subtract(a, b) : ii_multiply(a, factor);
    if (direction != 2) result = ii_rounded(result, exponent, direction);
    ii_store(output + tid * 5, result);
    closed[tid] = ii_closes(result, b) ? 1u : 0u;
}
"""


_ORIGINAL_SOURCE = gpu_runtime._source


def test_metal_multiword_primitives_match_checked_exact_integer_model(monkeypatch) -> None:
    """Actual Metal preserves carries, signed comparisons, invalidity and FP64 padding."""
    monkeypatch.setattr(gpu_runtime, "_source", _probe_source)
    runtime = gpu_runtime.ComputeRuntime("metal")
    rng = random.Random(123)
    edges = [0, 1, -1, 2**31, -2**31, 2**32 - 1, 2**32, -2**32, LIMIT, -LIMIT]
    rows, first, second, expected, closed = [], [], [], [], []
    pairs = [(a, b) for a in edges for b in edges]
    pairs += [(rng.randrange(-LIMIT, LIMIT), rng.randrange(-LIMIT, LIMIT)) for _ in range(400)]
    for a, b in pairs:
        for op in (0, 1, 2):
            for direction in (-1, 0, 1, None):
                exponent = rng.choice((-1074, -40, 0, 960, 1023))
                scale = IntegerScale(exponent)
                left = IntegerInterval.checked(a, min(LIMIT, a + rng.randrange(5)))
                right = IntegerInterval.checked(b, min(LIMIT, b + rng.randrange(5)))
                if rng.randrange(23) == 0:
                    left = IntegerInterval()
                factor = rng.choice((0, 1, 2, 3, 31, WORD_MASK))
                if op == 0:
                    result = scale.add(left, right, direction=direction)
                elif op == 1:
                    result = scale.subtract(left, right, direction=direction)
                else:
                    result = scale.multiply(left, factor, direction=direction)
                rows.append((op, exponent & WORD_MASK, factor,
                             2 if direction is None else direction & WORD_MASK))
                first.append(left.packed())
                second.append(right.packed())
                expected.append(result.packed())
                closed.append(result.valid and right.valid and result.upper <= right.lower)
    control, first, second, expected = [np.asarray(x, np.uint32)
                                       for x in (rows, first, second, expected)]
    actual, actual_closed = runtime.run(
        "bound_roots", len(rows),
        (control, first, second, np.empty_like(expected), np.empty(len(rows), np.uint32)), (3, 4),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_closed, closed)
    assert runtime.dispatches == 1 and runtime.counters["waits"] == 1


def test_cuda_generated_interface_compiles_with_stubbed_intrinsics(tmp_path) -> None:
    """The same integer body preserves CUDA C++ syntax; this is not hardware validation."""
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    stub = """
#include <algorithm>
using std::min; using std::max;
#define __device__
#define __forceinline__ inline
#define __global__
struct Dim { unsigned int x; };
Dim blockIdx, blockDim, threadIdx;
inline int atomicAdd(int* p, int v) { int old=*p; *p+=v; return old; }
inline int atomicCAS(int* p, int old, int v) { int x=*p; if(x==old)*p=v; return x; }
"""
    path = tmp_path / "integer_cuda.cpp"
    from freespace_sim.planner.colgen.gpu_integer_source import extend_source

    path.write_text(stub + extend_source(_ORIGINAL_SOURCE("cuda"), "cuda", gpu_runtime._ARGUMENTS))
    result = subprocess.run([compiler, "-std=c++17", "-fsyntax-only", str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("audit", [False, True])
def test_integer_search_covers_exhaustive_optimum_without_cpu_prune_verifier(
    monkeypatch: pytest.MonkeyPatch, signed: bool, audit: bool,
) -> None:
    """Exact signed unions and canonical routes certify while the original pruner is guarded.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forbids full CPU search and ordinary per-prune verification.
    - signed (bool): includes all negative credits and origin/visit overlap.
    - audit (bool): independently compares device deletions with original FP64 bounds.

    Return
    --------
    - output (None): actual Metal matches exhaustive canonical completion within certificate tolerance.
    """
    from freespace_sim.planner.colgen import pricing
    from freespace_sim.planner.colgen.gpu_pricing import _Batch
    from freespace_sim.planner.colgen.pricing_platform import PricingOptions, PricingPlatform
    from tests.test_colgen_pricing_platform import _assert_canonical, _oracle, _problem

    problem = _problem(signed=signed)
    if signed:
        # Keep every negative row while allowing enough deletions to exercise the audit.
        problem = replace(problem, duals={row: price * 0.001 if price < 0 else price
                                          for row, price in problem.duals.items()})
    optimum = _oracle(problem)

    def forbidden(*args, **kwargs):
        """Fail instead of using the original CPU search or per-prune numerical verifier."""
        pytest.fail("Original CPU pricing/prune verification was called")

    for name in ("price_flight", "_best_column", "_best_column_compiled"):
        monkeypatch.setattr(pricing, name, forbidden)
    if not audit:
        monkeypatch.setattr(_Batch, "bound", forbidden)
        monkeypatch.setattr(_Batch, "retain", forbidden)
    platform = PricingPlatform(PricingOptions(device="gpu", gpu_backend="metal",
                                               gpu_integer_bounds=True, gpu_integer_audit=audit))
    (result,) = platform.price_batch([problem])
    assert result.certified and result.search_complete
    assert result.reduced_cost == pytest.approx(optimum, abs=1e-6)
    assert result.upper_bound >= optimum - 1e-9
    assert result.upper_bound - result.reduced_cost <= 1.001e-6
    assert result.stats["candidates"] > 0
    if audit:
        assert result.stats["integer_audited"] > 0
    else:
        assert result.stats["cpu_recomputations"] == 0
    _assert_canonical(problem, result)
    # A fresh snapshot must observe new support and sign changes in exact membership tables.
    changed = replace(problem, duals={row: -price + 0.125 for row, price in problem.duals.items()})
    (again,) = platform.price_batch([changed])
    assert again.certified
    _assert_canonical(changed, again)


@pytest.mark.parametrize("max_labels", [1, 250_000])
def test_unrepresentable_operands_retain_alternatives_and_caps_remain_incomplete(
    monkeypatch: pytest.MonkeyPatch, max_labels: int,
) -> None:
    """An unusable grid can disable pruning but cannot erase an unresolved route.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies a deliberately inadequate grid and guards CPU search.
    - max_labels (int): covers a hard cap and complete enumeration with the same invalid data.

    Return
    --------
    - output (None): only the complete domain certifies; no invalid integer authorizes deletion.
    """
    from freespace_sim.planner.colgen import pricing
    from freespace_sim.planner.colgen.gpu_pricing import _Batch
    from tests.test_colgen_pricing_platform import _assert_canonical, _oracle, _platform, _problem

    problem = _problem()
    optimum = _oracle(problem)

    def forbidden(*args, **kwargs):
        """Refuse CPU fallback even when every integer operand is unavailable."""
        pytest.fail("Invalid numeric data called CPU search/prune verification")

    for name in ("price_flight", "_best_column", "_best_column_compiled"):
        monkeypatch.setattr(pricing, name, forbidden)
    monkeypatch.setattr(_Batch, "bound", forbidden)
    monkeypatch.setattr(_Batch, "bootstrap", lambda self: None)
    monkeypatch.setattr(IntegerScale, "for_values", classmethod(lambda cls, values, **kw: cls(-1074)))
    (result,) = _platform("metal", max_labels=max_labels, integer_bounds=True).price_batch([problem])
    assert result.stats["gpu_direct_prunes"] == 0
    assert result.certified == (max_labels > 1)
    assert result.search_complete == result.certified
    if result.certified:
        assert result.reduced_cost == pytest.approx(optimum, abs=1e-6)
        _assert_canonical(problem, result)
    else:
        assert result.status == "label_limit" and result.upper_bound is None


@pytest.mark.parametrize("options", [
    {"gpu_integer_bounds": 1}, {"gpu_integer_audit": 1}, {"gpu_integer_audit": True},
    {"gpu_integer_bounds": True, "cpu_recompute": False},
    {"gpu_integer_bounds": True, "gpu_prune_epsilon": 0.},
    {"gpu_integer_bounds": True, "device": "cpu", "gpu_backend": None},
])
def test_invalid_integer_certificate_modes_fail_before_search(options: dict) -> None:
    """Integer certification cannot inherit unchecked canonical scores or epsilon deletions."""
    from freespace_sim.planner.colgen.pricing_platform import PricingOptions

    with pytest.raises((TypeError, ValueError)):
        PricingOptions(**({"device": "gpu", "gpu_backend": "metal"} | options))
    if "gpu_prune_epsilon" not in options:
        from freespace_sim.planner.colgen.params import ColGenParams

        with pytest.raises((TypeError, ValueError)):
            ColGenParams(**{"pricing_" + name: value for name, value in
                            ({"device": "gpu", "gpu_backend": "metal"} | options).items()})
