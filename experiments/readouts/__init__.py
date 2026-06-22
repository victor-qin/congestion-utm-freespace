"""Readouts — standalone consumers of persisted runs.

Each module here is a small CLI that *loads* what the execute step (``experiments.run``) wrote and
emits an artifact, without ever re-simulating:

- per-run (a run folder, via ``runs.load_run``): ``replay``, ``figures``, ``uss_breakdown``
- cross-run (the shared ``index.parquet``, via ``runs.load_index``): ``curve``, ``compare``

This keeps analysis decoupled from execution — you can add a new readout, or re-slice old data, with
no new simulation.
"""
