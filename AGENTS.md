## Conversation
- Limit the usage of terms to describe code or functionality of features that you have not defined to the user. Work with the user to come up with terms.
- Be smart with token usage. Use subagents with lower reasoning where it won't affect quality of responses, such as for diagnostics and profiling work.

## Planning and Testing
- When writing or planning for a new feature, comprehensively clarify uncertainty in the functionality of the feature with the user.
- Write or plan clearly the tests that are clearly linked back to the functionality of a feature.
- Prefer extending or parametrizing existing tests over adding new tests. A few strong tests beat many narrow ones.
- Every test must trace to a functional requirement; avoid writing tests that pin implementation details or duplicate existing coverage.
- When running the test suite, run limited sets of tests that pertain to the changes only; run the whole suite only when it's needed, like on PRs before merge.

## Editing files

- Make the smallest safe change that solves the issue.
- Preserve existing style and conventions.
- Prefer patch-style edits (small, reviewable diffs) over full-file rewrites.
- After making changes, run the project’s standard checks when feasible (format/lint, unit tests, build/typecheck).

## Documentation (REQUIRED)
- For major functions (both public and private), the docstring should be standardized as:
```
"""
[Description]

Parameters
------------
- parameter (Type): description

Return
--------
- output (Type): description
"""
```
- For classes, the docstring should be standardized as:
```
"""
[Description]

Attributes
------------
- attribute (Type): description

"""
```
- Classes should include a description at the top before parameters are defined.
- Major functions should annotate every parameter and the return type in the signature
- Minor functions include at least a sentence long description, depending on how complex the function is.
- Comments should succintly explain why non-obvious code is the way it is, or state a constraint future edits must preserve (ordering, parity, byte-exactness). Comments do not serve as changelog.
- Avoid comments that try to describe the spatial or temporal functionality of a function, where a figure would be more illustrative. Instead, make a PNG and place it in `/context/figures/` and cite that figure as a comment.
- Put the code for figures in `/context/figures/make_figures.py`. Try to reuse code as much as possible.

## Reiviewing Pull Requests
- Prioritize bugs and flaws that affect the functionality of the changes and surface them to the user. If they arise from a lack of specification about functionality, ask for user feedback.
- Include in your review any ways to streamline or simplify code, without losing functionality.
- Whenever fixing code after a review, double-check the issues that were surfaced.
- Look for sections of code that can be removed.

## CONTINUITY.md (REQUIRED)

Maintain a single continuity file for the current workspace: `.agent/CONTINUITY.md`.
- `.agent/CONTINUITY.md` is a living document and canonical briefing designed to survive compaction; do not rely on earlier chat/tool output unless it's reflected there.
- At the start of each assistant turn: read `.agent/CONTINUITY.md` before acting.

### File Format

Update `.agent/CONTINUITY.md` only when there is a meaningful delta in:

  - `[PLANS]`: "Plans Log" is a guide for the next contributor as much as checklists for you.
  - `[DECISIONS]`: "Decisions Log" is used to record all decisions made.
  - `[PROGRESS]`: "Progress Log" is used to record course changes mid-implementation, documenting why and reflecting upon the implications.
  - `[DISCOVERIES]`: "Discoveries Log" is for when when you discover optimizer behavior, performance tradeoffs, unexpected bugs, or inverse/unapply semantics that shaped your approach, capture those observations with short evidence snippets (test output is ideal).
  - `[OUTCOMES]`: "Outcomes Log" is used at completion of a major task or the full plan, summarizing what was achieved, what remains, and lessons learned.

### Anti-drift / anti-bloat rules

- Facts only, no transcripts, no raw logs.
- Every entry must include:
  - a date in ISO timestamp (e.g., `2026-01-13T09:42Z`)
  - a provenance tag: `[USER]`, `[CODE]`, `[TOOL]`, `[ASSUMPTION]`
  - If unknown, write `UNCONFIRMED` (never guess). If something changes, supersede it explicitly (don't silently rewrite history).
- Keep the file bounded, short and high-signal (anti-bloat).
- If sections begin to become bloated, compress older items into milestone (`[MILESTONE]`) bullets.


## HISTORY.md (REQUIRED)

`context/HISTORY.md` is a durable record of changes or pitfalls that is preserved through pull requests. Follow the format and rules of `CONTINUITY.md` when writing to it.

There is an extremely high-bar for transferring things from `.agent/CONTINUITY.md` to `context/HISTORY.md`.
- Only changes that are used to record mistakes that are likely to recur between PRs should be recorded
- In addition to following the format of `CONTINUITY.md`, note the file(s) that are relevant to the recorded comment.
- `HISTORY.md` will only be updated when a PR is created, or when it is merged.

## Repository
- Files in `context/` and `analysis/` do not need to be maintained.
- If reusing a script from `context/` and `analysis/`, double check for correcteness before using them, especially with files in `freespace_sim/`.

## SuperCloud (remote runs)
- Prefer this machine for small runs: tests, debugging, profiling, quick benchmarks, and any single run that finishes in under ~30 min and fits in this machine's memory (128 GB, 20 cores). Local runs start immediately. SuperCloud jobs wait in a queue and recompile numba from scratch, so small runs are slower there.
- Send big runs to MIT SuperCloud: a run that would take more than ~30 min here, needs more memory than this machine has, or needs CUDA (V100), and batches of independent runs (seeds, sweeps) that would otherwise run one after another here.
- One tool does it all: `uv run python -m experiments.supercloud {doctor,submit,status,fetch,cancel,list}`. `submit` ships your worktree exactly as it is on disk (uncommitted edits included) and keeps that code as `refs/supercloud/<run_id>`; nothing is committed to your branch. Gitignored files (`results/`, `*.parquet`, …) are not shipped, and a file over 20 MB stops the submit.
- Submit: `uv run python -m experiments.supercloud submit --name <label> --cores N --hours H [--gpus G] [--gurobi] [--env NAME=VALUE] -- experiments.run ... --no-progress`. Only `experiments.run`, `pytest` and `experiments.colgen_four_round` (fixed-input per-round colgen profile; its `--source` option is refused so the job always runs the shipped snapshot) may run.
- Cores also buy memory (xeon-p8: 4 GB per core); size runs with the table in `experiments/batch/supercloud_array.sh`. GPU jobs: about 20 cores per GPU.
- Filing jobs, per the SuperCloud docs ([submitting jobs](https://mit-supercloud.github.io/supercloud-docs/submitting-jobs/), [filesystem](https://mit-supercloud.github.io/supercloud-docs/filesystem/), [limits](https://mit-supercloud.github.io/supercloud-docs/resource-limit-enforcement-and-increases/)):
  - Memory comes only with cores ("request more cores"). Size from a measured run: `sstat -j <job>.batch --format=MaxRSS` while it runs, `sacct -j <job> -o JobID,AllocCPUS,MaxRSS --units=G` after.
  - Timing and benchmark runs need the whole node; the docs ask for exclusive mode for performance tests. Here that means `--cores 48` on xeon-p8, so no other job shares the node's memory bandwidth. Runs whose timing doesn't matter request only what they need.
  - Keep concurrently busy processes and threads at or below `--cores`. The job pins OMP/BLAS/numba threads to 1; set worker and solver thread counts explicitly. A colgen pricing pool and Gurobi take turns, so each may approach the core count on its own.
  - Prefer fewer, larger jobs over many small ones. Batch-job wall limit: 4 days. Heavy intermediate I/O goes to `$TMPDIR` (node-local, deleted at job end); keep files ≥1 MB and under ~1,000 per directory.
  - `--env` may not set `PATH`, `HOME`, `GRB_*` (use `--gurobi`), `LD_*`, `DYLD_*`, or the `PYTHON*` variables that change which code loads (`PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, …). Others are fine, e.g. `--env PYTHONHASHSEED=0`.
- `--gurobi` uses SuperCloud's full academic licence; pair it with `--colgen-solver gurobi`, and set solver threads with `--env COLGEN_GUROBI_THREADS=N`.
- Caps are in `~/.config/congestion-utm/supercloud.toml` and belong to the user. Never edit that file; if a submit is refused, ask the user.
- `fetch <run_id>` copies results and the Slurm log to the main checkout's `results/supercloud/<run_id>/`. Cite remote results by run id.
- `submit` and `doctor --gpu` can take minutes (lock wait, upload, GPU queue): give them a tool timeout of at least 15 minutes. A submit queued a job only if it printed `submitted run … as job …`. After an interrupted or failed submit, check `list` and `status` before retrying, and run `doctor` (add `--gpu` to test a V100 node) if the cause is unclear.
- Codex agents need network access to `txe1-login.mit.edu`, read access to `~/.ssh`, and write access to the main checkout's `.git/` (snapshot refs), its `results/supercloud/` (ledger) and `~/.config/congestion-utm/` (submit lock).
