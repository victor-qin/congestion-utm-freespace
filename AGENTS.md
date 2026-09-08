

## Planning and Testing
- When writing or planning for a new feature, comprehensively clarify uncertainty in the functionality of the feature with the user.
- Write or plan clearly the tests that are clearly linked back to the functionality of a feature.
- Prefer extending or parametrizing existing tests over adding new tests. A few strong tests beat many narrow ones.
- Every test must trace to a functional requirement; delete tests that only pin an implementation detail or duplicate existing coverage.

## Editing files

- Make the smallest safe change that solves the issue.
- Preserve existing style and conventions.
- Prefer patch-style edits (small, reviewable diffs) over full-file rewrites.
- After making changes, run the project’s standard checks when feasible (format/lint, unit tests, build/typecheck).

## Documentation (REQUIRED)
- For major functions, the docstring should be standardized as:
```
***
[Description]

Parameters
------------
- parameter (Type): description

Return
--------
- output (Type): description
***
```
- Classes should include a description at the top before parameters are defined.
- Minor functions include at least a sentence long description, depending on how complex the function is.
- Comments should succintly explain why non-obvious code is the way it is, or state a constraint future edits must preserve (ordering, parity, byte-exactness). Comments do not serve as changelog

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


