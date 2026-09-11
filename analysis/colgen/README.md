# Column-generation benchmark tools

These modules preserve the benchmark and validation workflow used for PR 133. Run them
from the repository root with the project environment. Every command that loads
`freespace_sim` accepts an explicit source tree and records hashes of that tree and its
inputs. Output directories must be fresh.

## Full benchmark

`benchmark` runs each arm in a fresh process and keeps simulations serial. For a small
two-tree comparison:

```sh
uv run python -m analysis.colgen.benchmark run \
  --baseline /path/to/baseline --fixed /path/to/candidate \
  --scenarios colgen_test --seeds 0 --iterations 2 \
  --workers 2 --out /private/tmp/colgen-pair
```

The validated 1,200-second outbound policy, with 12 pricing workers and a 30-second IP
solve after every round, can be captured from one selected source tree with:

```sh
uv run python -m analysis.colgen.capture \
  --source-root /path/to/source --out /private/tmp/colgen-capture \
  --rounds 1 6 11 --expected-production-flights 1537 --allow-missing-rounds -- \
  --version iteration_ip_eager --scenario density_faa_wing_zipline \
  --seed 0 --flights 0 --iterations 30 --budget 14460 \
  --ip-budget 600 --iteration-ip-budget 30 --ip-reserve 660 \
  --capture-columns --overrun 6 --columns-per-flight 1 --bootstrap-method astar \
  --workers 12 --gurobi-threads 4 --lp-gap 0.001 \
  --exact-pricing-interval 5 --destroy 100 --gap 0 \
  --demand-seconds 1200 --outbound-only
```

Use `--benchmark /path/to/benchmark.py` to capture with an archived benchmark file, or
`--benchmark-module package.module` to select another importable implementation. The
packaged `analysis.colgen.benchmark` module is the default.

## Captured sweep replay

Replay one captured pricing round in a fresh process. Sweep timing includes worker startup
and worker graph setup. Parent acceptance and independent cold claim, cost, and reduced-cost
validation are timed separately.

```sh
uv run python -m analysis.colgen.replay \
  --source-root /path/to/source \
  --capture /private/tmp/colgen-capture/pricing_captures/round_0001.pkl \
  --out /private/tmp/colgen-replay --budget 1800 \
  --expected-workers 12 --gurobi-threads 4 \
  --require-independent-clean --require-all-parent-certified
```

Pass `--reference /path/to/reference-replay` to require output, identity, claim, cost,
reduced-cost, and search-work parity with an earlier replay.

## Capacity row scan

The row harness measures public capacity separation over the one-column-per-flight
incumbent in a capture. Pool construction and the fractional-load control are reported
outside the measured row scan.

```sh
uv run python -m analysis.colgen.rows \
  --source-root /path/to/source \
  --capture /private/tmp/colgen-capture/pricing_captures/round_0001.pkl \
  --out /private/tmp/colgen-rows --repeats 9
```

`analysis.colgen.trace.ColumnTrace` is an instrumentation helper used by
`benchmark --capture-columns`; it is not a standalone command. `capture` stores trusted local pickle
artifacts, so only replay captures produced by a source you trust.
