# Capacity separation and certified pricing benchmark

Start with [the final report](final_report.md) and
[the two-seed comparison](full_comparison/report.md). The baseline is profiling
branch commit `40c25e9ea63165eda6e173c4095d31e1e903dede`; the combined V2 source
adds the capacity-scan and worker-certification changes. This comparison does
not measure the complete profiling branch against main.

The committed evidence includes source/test manifests, aggregate measurements,
per-seed parity audits, component replay summaries, the 25-second IP policy
comparison, and the scripts used to run and summarize the experiments. Raw
captures, full flight traces, logs, source archives, and incomplete attempts
remain local under this directory and are not included in the PR. References to
those files in reports are provenance locations, not promised Git-tracked files.
The final PR regression scope, results, and tested source hashes are recorded in
[`pr_validation.json`](pr_validation.json).

`run_full_matrix.py`, `run_ablations.py`, and `run_policy25.py` are archived
experiment drivers. They expect explicit frozen source trees and some original
temporary paths; review their arguments before reusing them. The baseline can
be reconstructed from the commit above, and the V2 production files are recorded
in `combined_v2_source_manifest.json`. `capture_full_run.py` wraps the archived
`benchmark_colgen_lns.py`, which uses its sibling `colgen_trace.py`.

PR review subsequently added a guard rejecting malformed primary sweep tuples
whose column flight IDs do not match their positional flight IDs. That guard and
its regression are newer than the measured V2 snapshot; valid pricing searches
are unchanged, and the full benchmarks were not rerun for this input check.

The configuration is a 1200-second FAA demand window filtered to outbound flights,
seeds 0 and 1, 12 pricing workers, 4 Gurobi threads, six extra hops, one column per
flight, hybrid A* plus restricted DP bootstrap, nominal plus 20 later seed
departures, a 0.1% LP target, at most 30 CG rounds, 30 seconds per round IP, and
600 seconds for the final IP with a 660-second reserve. These are benchmark
settings; in particular the 600-second final IP allowance is not a new default.

Each full comparison has one serial timing trial per arm and seed. Preserve the
distinction between identical column pools/costs and potentially different
finite-time IP selections, and between an optimal restricted IP and a global
integer optimality proof. Deferred pricing work is tracked in
[issue #132](https://github.com/victor-qin/congestion-utm-freespace/issues/132).
