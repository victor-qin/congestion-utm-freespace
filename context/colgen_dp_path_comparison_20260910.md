# Native DP path comparison

Work is on `codex/colgen-ip-profiling`. The before-change source is frozen at
`/private/tmp/colgen-pathcmp-baseline-20260910`, from commit `6a4c815`.

## Change and exactness

The native pricing DP resolves equal scores with the same deterministic key as the
Python reference: `(hops, departure, origin_lane, path)`. The old path comparison
followed both parent chains, wrote and reversed both complete paths, then compared
their cells from the root. Profiling found this work was frequent, contrary to its old
docstring's assertion that it was outside the hot path.

The new `_path_cmp_equal_depth` handles the equal-hop case reached by `_tie_lt`.
It visits the two parent chains together, starting at their leaves, retaining the
comparison at every unequal pair of cells. Each earlier difference replaces a later
one. If the walks meet at the same label ID, their remaining parent chains are
identical and the walk stops. Equal cell values alone do not establish shared ancestry.

For paths $X=(x_0,\ldots,x_H)$ and $Y=(y_0,\ldots,y_H)$, tuple order is determined by

$$k=\min\{i:x_i\ne y_i\},\qquad \operatorname{cmp}(X,Y)=\operatorname{sgn}(x_k-y_k).$$

The backward walk visits differing indices in decreasing order, so its final retained
comparison is exactly the one at $k$. If there is no differing index it returns zero.
Both chains have equal length because roots have zero hops and every parent removes
one hop. Accepted ancestors are immutable. This also handles identical paths represented
by different label IDs, without hashing or approximate equality.

The change retains all score tolerances, root/arc insertion order, dominance decisions,
and sink certification rules. It adds no memory per label. Work is proportional to the
unshared suffix; worst-case complexity is still linear in path length. Generic path
comparisons in the feasible-search frontier retain the existing unequal-length and
prefix handling.

## Validation protocol

- Compare both supported comparator modes with Python tuple ordering over randomized
  label trees, including the general comparator's shorter-prefix rule.
- Exercise shared ancestors, merely shared cells, duplicate roots, opposing early/late
  differences, identical labels, deep trees, and reuse of an unpublished tentative slot.
- Run native/reference dominance, layer-order, sink-order and complete-pricing checks,
  including epsilon ties, different hop limits, bootstrap modes and forbidden rows.
- Replay the same six real captured flights with the current six-extra-hop and hybrid
  bootstrap settings. Separate compilation from repeated warm native and whole-flight
  timing, and require exact returned columns, claims, costs, reduced costs and work counts.
- Replay all 1,537 captured outbound flights with 12 workers, sequentially for each
  version, preserving raw diagnostics and comparing exact results.

All **159 targeted tests passed** in 463.71 seconds. The only warning came from the
intentional worker-death test. Lint and whitespace checks passed. Evidence:
`analysis/colgen_pathcmp_20260910/{regression.xml,validation.json}`.

## Matched flight timings

Each row is the median of three warm replays, after an untimed first solve. These use
today's hybrid bootstrap and six-extra-hop settings, so they should not be compared
directly to the older profiler's pre-bootstrap flight timings.

| Flight | Total before/after, s | Native DP before/after, s | Native reduction |
|---|---:|---:|---:|
| 934 | 0.754 / 0.629 | 0.291 / 0.190 | 34.7% |
| 2692 | 0.719 / 0.594 | 0.293 / 0.193 | 34.1% |
| 1520 | 0.308 / 0.284 | 0.094 / 0.080 | 14.6% |
| 1600 | 1.075 / 0.937 | 0.433 / 0.353 | 18.5% |
| 1640 | 2.484 / 1.658 | 2.084 / 1.287 | 38.2% |
| 2282 | 3.948 / 2.507 | 3.408 / 1.972 | 42.1% |

All six returned outputs, timed path identities, capacity claims, cost/reduced-cost
float bits, and search work counts matched the frozen baseline exactly. Label-array
capacity was unchanged. Evidence: `analysis/colgen_pathcmp_20260910/candidate_single`.

## Complete pricing sweep

The captured first-LP sweep contains 1,537 flights from the 1200-second FAA outbound
scenario. Both versions use 12 workers, six extra hops, one column per flight, and the
current hybrid bootstrap. Runs were sequential, in baseline/candidate/candidate/baseline
order; tests and instrumented runs never overlapped timed runs.

| Version | Two sweep times, s | Median, s |
|---|---:|---:|
| Baseline | 88.867, 87.593 | 88.230 |
| New comparison | 83.260, 82.939 | 83.100 |

The observed median pricing time fell **5.81%**. Every repeated sweep matched all
1,537 baseline outputs, route identities, claims, cost/reduced-cost bits, and recorded
search work exactly. All sweeps completed with no fallbacks, declines, or worker restarts.
Two samples per version establish the observed range, not broad statistical confidence.
These times include pricing worker startup, preparation, and warmup; master/IP solving
was not rerun, so this is not a measured end-to-end CG + IP improvement.

Separate instrumentation across the six selected flights recorded **808,169,976 path
nodes materialized before, zero after** during native DP ties. Comparison call counts
and every non-path native work counter remained identical. The new comparator visited
300,735,894 parent pairs; parent pairs and materialized nodes are different units, so
their ratio is not a runtime attribution. Label-array capacity was unchanged; whole
process RSS was not measured.

Detailed evidence and reproduction commands are in
[the benchmark report](../analysis/colgen_pathcmp_20260910/report.md).
Both measured source trees are archived beside the report; `source_delta.json` confirms
that only `dp_kernel.py` differs. The general feasible-search comparator and extraction
of an actual winning path still reconstruct paths when needed.
