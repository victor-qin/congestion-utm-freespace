"""Configuration for the column-generation planner."""

from __future__ import annotations

import math
import operator
import os
from dataclasses import dataclass


# Planners whose schedule colgen can translate into a warm start. A module constant so
# `params` can validate a name without importing the planner registry: colgen is imported BY
# the sim, so reaching back for it here would be circular.
WARM_START_PLANNERS = frozenset({"astar"})


@dataclass(frozen=True, slots=True)
class ColGenParams:
    """Network and solver controls for one column-generation run.

    The gaps use relative objective units (``1e-4`` means 0.01%).  ``M`` is the
    per-flight benefit in the maximize master objective ``M - delay_s``; it is
    deliberately much larger than the shipped ground-delay budget so a usable
    trajectory dominates cancellation.  ``time_limit_s`` is a best-effort
    whole-solve wall budget: pricing and native LP/IP calls receive the remaining
    time, with checks between graph, seed, and lazy-separation rounds.  A single
    synchronous geometry/backend call cannot be preempted and may overrun slightly.
    """

    # THE knob that sizes the pricing search, and it is one degree of freedom doing two jobs:
    # how far over the LATTICE geodesic a priced route may fly (in hops; one hop = `dt_s` of air
    # time), and the spatial corridor derived from that budget. Terminal-lane corridors
    # account for their fixed prefixes/suffixes and envelope the remaining endpoint paths;
    # the corridor is derived, not an independent choice.
    #
    # ABSOLUTE, not a fraction of the flight length: every flight gets the same room to route
    # around a busy cell regardless of distance, where a fractional cap is harshest exactly
    # where the absolute room is smallest. Measured against the lattice geodesic, not
    # `enroute_reference_m`, because hex quantization alone already puts most flights well over
    # the Euclidean straight line.
    #
    # 0 pins the geodesic; a large value effectively disables the bound. Deliberately
    # suboptimal -- a route needing more than this is unreachable even if optimal -- so this is
    # the first knob to widen (`--colgen-max-air-overrun`) if a congested scenario denies
    # flights pricing ought to place, and the dominant term in how much search a sweep does.
    max_air_overrun_hops: int = 6
    # Gurobi by default, NOT "auto": "auto" falls back to HiGHS silently when gurobipy is
    # missing, and the backend is answer-affecting (Gurobi's duals can close the revenue gap at
    # iteration 1 where HiGHS runs many more). "gurobi" raises instead, naming the missing
    # extra. Gurobi also scales -- HiGHS reaches the master through `scipy.optimize.milp`, which
    # takes no incumbent, so its final IP runs cold and single-threaded. Colgen therefore needs
    # `uv sync --extra gurobi` and a real licence; set "highs" to opt out, or "auto" for the old
    # silent fall-back.
    solver: str = "gurobi"
    max_iterations: int = 30
    # Flights released per LNS try; zero retains the rounding heuristic.
    lns_destroy_flights: int = 0
    # Before sequential swaps, jointly repair the most LP-disagreed neighborhood.
    # Budget includes construction; 0 retains the original sequential-only heuristic.
    lns_joint_time_limit_s: float = 0.5
    # Default: replace rounding/LNS with a full-pool IP after each completed pricing round.
    # Zero disables it. Each call is bounded by the pricing deadline so the final
    # IP reserve remains available; feasible improvements seed the next round.
    iteration_ip_time_limit_s: float = 30.0
    iteration_ip_eager: bool = True
    # Best-effort whole-solve wall budget (20 min). The old 120 s default could not finish a
    # single pricing sweep on a real instance and reported `time_limit` with a heuristic-only
    # schedule. `ip_reserve_s = min(5, 0.05 * t)` is already at its cap here, so the tail left
    # for the final IP does not move with this; an enabled greedy's rate is still clamped by
    # `pricing_deadline`, so raising this lifts that stage's ceiling on a large enough batch.
    time_limit_s: float = 1200.0
    # Caps the FINAL restricted-master IP on its own, NOT as a share of `time_limit_s`. Without
    # it a loop that converges early (on `lp_gap` or `max_iterations`) hands the MILP every
    # remaining second -- unbounded from the IP's point of view precisely when the loop went
    # well. Exceeding it is not a failure: `solve_ip` falls back to the independently validated
    # rounding incumbent and reports a non-optimal `ip_status`, so the run still produces a
    # claim-feasible (just uncertified) schedule. Composes with `ip_reserve_s`: that is how much
    # pricing holds back, this is how long the IP may then run.
    ip_time_limit_s: float = 120.0
    # Whole-solve seconds held back from preprocessing, LPs and pricing for the final IP
    # stage. None preserves the legacy min(5, 5% of time_limit_s) reserve. This includes
    # IP setup and separation, so reserve more than ip_time_limit_s when the full native
    # search allowance is required. The whole-solve deadline remains authoritative.
    ip_reserve_s: float | None = None
    # Ceiling on rows the final IP may pre-materialize, or None for no ceiling (the shipped
    # default, and the behaviour measured throughout this PR).
    # `RestrictedMaster.materialize_bindable_rows` is all-or-nothing: over the bound it
    # materializes NOTHING and the separation loop runs as before, because a partial set is the
    # worst of both. Worth having reachable because eager materialization scales WITH the pool,
    # so a denser pool than any measured here would otherwise have no brake.
    max_eager_ip_rows: int | None = None
    lp_gap: float = 1e-3
    ip_gap: float = 1e-3
    # Revenue per served flight in the set-packing objective `sum (M - delay_s) x`. Its only
    # requirement is that serving beats denying (`M > max delay_s`); everything above that is
    # numerical harm, because the coefficients `M - delay_s` then carry the optimized quantity
    # in their last few digits. The delay bound is ~4,600 s (ground capped by
    # `max_ground_delay_s`, air by `max_air_overrun_hops`), so 1e4 clears it ~2.2x. Reported
    # cost is `n*M - ip_objective` and `Column.delay_s` has no M in it, so archived numbers stay
    # on one ruler -- but M conditions the search, so it is answer-affecting even though the
    # metric is not.
    M: float = 10_000.0
    epsilon: float = 1e-6
    # Two consumers that scale differently. `master.round_heuristic` runs this many randomized
    # rounding restarts per iteration (cheap; 16/32/64 returned the identical incumbent on 98
    # flights, so the low default is honest). `_greedy_feasible_selection` ALSO derives its
    # candidate cap from this (x16), unrelated to rounding -- worth knowing before changing it,
    # since that stage walks flights in order and the cap truncates how many it reaches.
    n_heuristic_tries: int = 16
    # What the master minimises. `total_cost` weights ground:air 1:3 to match the config dials
    # (`cost_ground_delay_per_s`, `cost_air_lateral_per_s`); `total_delay` weights them equally,
    # which is degenerate: `ground + flown` is invariant under a ground-for-air swap, so many
    # columns tie EXACTLY and the label DP's dominance cannot separate labels the real objective
    # strictly orders. total_cost also prunes far more (its completion envelope is ~3x shorter),
    # so benchmarking pricing under total_delay measures the weakest pruning regime, not the
    # shipped one. See pricing.py.
    objective: str = "total_cost"
    # Which scale the lp_gap / ip_gap thresholds use.
    #   "revenue" -- the paper's eqs (10)/(11): (UB - RMP)/RMP on the maximize objective, whose
    #                scale includes n*M.
    #   "cost"    -- the same absolute gap normalized by total cost; stricter when M >> cost.
    # Pinned to "cost" because "revenue" is not scale-free in M: it reduces to `tau*M` (n
    # cancels), so it reads "stop unless the average flight can still save `tau*M` seconds" --
    # 100 s at M=1e6 but 1 s at M=1e4. "cost" measures the gap against the quantity being
    # optimized and does not move with M, which is the property a default needs.
    gap_metric: str = "cost"
    # Optionally run another planner first and hand colgen its schedule -- as pool columns AND
    # the starting incumbent. None (default) opens from colgen's own geodesic seeds; "astar"
    # seeds from FCFS A*. OFF BY DEFAULT for reporting honesty, not cost (A* is cheap against a
    # colgen solve): enabling it silently would make "colgen" in every future result mean "A*
    # plus colgen refinement" -- a different, weaker claim -- with no way to tell archived runs
    # apart. Reported in `stats` for the same reason.
    warm_start_planner: str | None = None
    # False starts from supplied columns only: no nominal-route construction,
    # departure ladder, or greedy nominal-route scheduling pass. Pricing may
    # still generate nominal routes later. The solve requires supplied columns.
    seed_nominal_routes: bool = True
    # For each supplied flight's first column, add up to this many departure
    # steps on EACH side, clipped to the graph's legal departure window.
    # Zero leaves provided columns unchanged; 10 offers up to 20 alternatives.
    provided_seed_ladder_steps: int = 0
    # Extra departure steps allowed when fitting translated warm-start routes.
    # Zero preserves translated departures; provided-only mode leaves joint
    # conflicts to the IP instead of filtering or repairing the imports.
    warm_start_max_shift_steps: int = 8
    # Pure clock translations of each flight's seed, offered to the master before the first LP.
    # A shift is arithmetic, not a search, and pricing otherwise spends its early iterations
    # rediscovering exactly these. See :func:`solver._add_departure_ladder`; 0 disables.
    seed_ladder_steps: int = 20
    # Spacing between departure alternatives in lattice steps.
    seed_ladder_stride: int = 1
    # Wall clock for the post-first-LP greedy, PER FLIGHT. NOW 0, WHICH DISABLES THE STAGE. The
    # stage's cutoff is measurably worthless: it produces `best_heuristic`, handed to pricing as
    # the `known_column` each subproblem prunes against, whose reduced cost `entry_rc` is exactly
    # 0.0 on every flight of every sweep measured -- LP duality forces it (pool columns have
    # rc <= 0 by optimality, complementary slackness pins the basic one at 0). Removing it leaves
    # pricing's time unchanged and moves the objective by ~0.1% with no consistent sign. Turning
    # it off does NOT remove the fallback schedule: `best_heuristic` is set from `round_heuristic`
    # BEFORE this stage, which only replaces it on a measured improvement. Most likely wrong where
    # `time_limit_s` genuinely binds and a better heuristic is the answer rather than a head start
    # -- an untested regime, so raise it there if it matters.
    greedy_budget_s_per_flight: float = 0.0
    # Worker processes for the per-iteration pricing sweep; 0 keeps it in-process. A pure
    # performance knob (`pricing_pool.price_sweep` reproduces the sequential loop's accepted
    # prefix and order) but NOT free: each worker rebuilds every graph and holds its own label
    # pool, so memory is linear in this count (~12.5 GB at 4 workers on 50 density flights),
    # which is what forecloses a 4 GB/core cluster node and keeps the default at 0. An
    # OOM-killed worker HANGS the sweep rather than raising (see `pricing_pool`), so the failure
    # mode is silence. Answer-identical to the sequential loop ONLY on a sweep that FINISHES:
    # `pricing_deadline` is a wall clock, so a pool keeps a longer accepted prefix within the
    # same budget -- a parity A/B must pin this to 0 on both arms. This is the single home for
    # the setting.
    n_pricing_workers: int = 0
    # `mp.Pool.imap`'s chunksize. 1 is right and raising it is a trap: a pricing task ships one
    # int in and ~14 KB out against tens of seconds of compute, so there is nothing to amortize,
    # while chunking costs load balance (pre-partitioned chunks leave nothing to rebalance a
    # straggler against). Kept configurable so the claim stays measurable.
    pricing_chunksize: int = 1
    # Experimental restricted pricing: search only bootstrap_roots promising starts.
    # Its reduced costs are achievable scores, never global bounds. Full pricing
    # runs every N rounds, on cheap stagnation, and on the last allowed round.
    cheap_pricing: bool = False
    exact_pricing_interval: int = 5
    # How many ROOTS the pricing BOOTSTRAP searches before the real search, in descending order
    # of `PreparedVariants.score`; 0 disables. A root is one `(departure_step, origin lane)`
    # pair. Ranked rather than truncated to a departure prefix, which would miss an optimum that
    # departs late. It exists because the cutoff pricing enters with is structurally ZERO, not
    # merely weak: every column pricing can reach for free is already in the master's pool, LP
    # optimality forces those rc <= 0, and complementary slackness pins the basic one at 0, so
    # the only route to a positive cutoff is to SEARCH for one. The bootstrap and the greedy are
    # SUBSTITUTES (both supply a `known_column`); with the greedy off this is the only cutoff
    # source. Optimality-safe -- pruning against a certified achievable score discards nothing
    # strictly better -- but under dominance a tighter cutoff can return a different
    # equally-optimal column, so treat it as answer-affecting and re-baseline parity.
    bootstrap_roots: int = 1
    # WHAT the bootstrap sorts roots on. "score" is `PreparedVariants.score` (cost-so-far, g);
    # "bound" is `completion_can_compete`'s `hop_rc_bound` at the root's minimum feasible hop
    # count (g + h, already computed by the root gate, so free to read). "bound" is what makes
    # `bootstrap_roots=1` viable: `score` carries one live bit ("depart earlier") and cannot
    # tell a short remaining route from a long one, where g + h can. Inert on single-lane
    # flights (identical ordering), decisive on multi-lane ones. Ordering only -- it cannot
    # prune, because the bootstrap returns an incumbent and the main search still fans over
    # every root.
    bootstrap_ranking: str = "bound"
    # A* supplies a quick candidate; restricted DP still refines it before full pricing.
    bootstrap_method: str = "astar"
    bootstrap_max_labels: int = 20000
    # Additional certified surviving sinks, not an exhaustive k-best route search.
    columns_per_flight: int = 1

    @property
    def effective_ip_reserve_s(self) -> float:
        return (min(5.0, 0.05 * self.time_limit_s) if self.ip_reserve_s is None
                else self.ip_reserve_s)

    def __post_init__(self) -> None:
        """Normalize and validate every field after construction, raising on bad input.

        Parameters
        ------------
        - none: reads the frozen dataclass fields and rewrites them in place.

        Return
        --------
        - output (None): coerces fields via ``object.__setattr__``; raises ``TypeError`` on a
          wrong-typed field or ``ValueError`` on an out-of-range one.
        """
        if self.bootstrap_method not in {"astar", "dp"}:
            raise ValueError("bootstrap_method must be 'astar' or 'dp'")
        for name in ("bootstrap_max_labels", "columns_per_flight"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            value = operator.index(value)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if isinstance(self.max_air_overrun_hops, bool):
            raise TypeError("max_air_overrun_hops must be an integer")
        try:
            overrun = operator.index(self.max_air_overrun_hops)
        except TypeError as exc:
            raise TypeError("max_air_overrun_hops must be an integer") from exc
        if overrun < 0:
            raise ValueError("max_air_overrun_hops must be non-negative")
        object.__setattr__(self, "max_air_overrun_hops", overrun)

        # Zero would make every IP a no-op that silently returns the rounding incumbent -- a
        # legitimate thing to want, but not one to reach by accident, so it must be a positive
        # number asked for explicitly.
        try:
            ip_limit = float(self.ip_time_limit_s)
        except (TypeError, ValueError) as exc:
            raise TypeError("ip_time_limit_s must be a number") from exc
        if not math.isfinite(ip_limit) or ip_limit <= 0.0:
            raise ValueError(
                f"ip_time_limit_s must be finite and positive, got {self.ip_time_limit_s!r}"
            )
        object.__setattr__(self, "ip_time_limit_s", ip_limit)

        for name in ("iteration_ip_eager", "cheap_pricing", "seed_nominal_routes"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if isinstance(self.warm_start_max_shift_steps, bool):
            raise TypeError("warm_start_max_shift_steps must be an integer")
        try:
            warm_shift = operator.index(self.warm_start_max_shift_steps)
        except TypeError as exc:
            raise TypeError("warm_start_max_shift_steps must be an integer") from exc
        if warm_shift < 0:
            raise ValueError("warm_start_max_shift_steps must be non-negative")
        object.__setattr__(self, "warm_start_max_shift_steps", warm_shift)
        if isinstance(self.provided_seed_ladder_steps, bool):
            raise TypeError("provided_seed_ladder_steps must be an integer")
        try:
            provided_steps = operator.index(self.provided_seed_ladder_steps)
        except TypeError as exc:
            raise TypeError("provided_seed_ladder_steps must be an integer") from exc
        if provided_steps < 0:
            raise ValueError("provided_seed_ladder_steps must be non-negative")
        object.__setattr__(self, "provided_seed_ladder_steps", provided_steps)
        interval = self.exact_pricing_interval
        if isinstance(interval, bool):
            raise TypeError("exact_pricing_interval must be an integer")
        interval = operator.index(interval)
        if interval < 1:
            raise ValueError("exact_pricing_interval must be positive")
        object.__setattr__(self, "exact_pricing_interval", interval)

        iteration_ip_limit = float(self.iteration_ip_time_limit_s)
        if not math.isfinite(iteration_ip_limit) or iteration_ip_limit < 0.0:
            raise ValueError("iteration_ip_time_limit_s must be finite and non-negative")
        object.__setattr__(self, "iteration_ip_time_limit_s", iteration_ip_limit)

        # None is "no ceiling", distinct from 0 ("never materialize eagerly", a way to pin the
        # old lazy separation loop for an A/B). Both are reachable, so reject only negatives and
        # non-integers.
        if self.max_eager_ip_rows is not None:
            if isinstance(self.max_eager_ip_rows, bool):
                raise TypeError("max_eager_ip_rows must be an integer or None")
            try:
                max_eager = operator.index(self.max_eager_ip_rows)
            except TypeError as exc:
                raise TypeError("max_eager_ip_rows must be an integer or None") from exc
            if max_eager < 0:
                raise ValueError(
                    f"max_eager_ip_rows must be non-negative, got {self.max_eager_ip_rows!r}"
                )
            object.__setattr__(self, "max_eager_ip_rows", max_eager)

        if not isinstance(self.solver, str):
            raise TypeError("solver must be a string")
        solver = self.solver.lower()
        if solver not in {"auto", "gurobi", "highs"}:
            raise ValueError("solver must be one of 'auto', 'gurobi', or 'highs'")
        object.__setattr__(self, "solver", solver)

        if not isinstance(self.objective, str):
            raise TypeError("objective must be a string")
        if self.objective not in {"total_delay", "total_cost"}:
            raise ValueError("objective must be 'total_delay' or 'total_cost'")
        if not isinstance(self.gap_metric, str):
            raise TypeError("gap_metric must be a string")
        if self.gap_metric not in {"revenue", "cost"}:
            raise ValueError("gap_metric must be 'revenue' or 'cost'")
        # Validated against a whitelist rather than resolved lazily: an unknown name would
        # otherwise surface as a silent no-op warm start, and "colgen ran unseeded" is
        # indistinguishable in the output from "colgen was seeded and it did not help".
        if self.warm_start_planner is not None:
            if not isinstance(self.warm_start_planner, str):
                raise TypeError("warm_start_planner must be a string or None")
            if self.warm_start_planner not in WARM_START_PLANNERS:
                raise ValueError(
                    f"warm_start_planner must be None or one of "
                    f"{sorted(WARM_START_PLANNERS)}, got {self.warm_start_planner!r}"
                )

        if isinstance(self.seed_ladder_steps, bool):
            raise TypeError("seed_ladder_steps must be an integer")
        try:
            ladder = operator.index(self.seed_ladder_steps)
        except TypeError as exc:
            raise TypeError("seed_ladder_steps must be an integer") from exc
        if ladder < 0:
            raise ValueError("seed_ladder_steps must be non-negative")
        object.__setattr__(self, "seed_ladder_steps", ladder)

        if isinstance(self.n_pricing_workers, bool):
            raise TypeError("n_pricing_workers must be an integer")
        try:
            workers = operator.index(self.n_pricing_workers)
        except TypeError as exc:
            raise TypeError("n_pricing_workers must be an integer") from exc
        if workers < 0:
            raise ValueError("n_pricing_workers must be non-negative")
        # An upper bound, because the failure past it is not an error message: each worker
        # rebuilds every graph and carries its own label pool (~1.5 GB on density), so an
        # over-large count from a config file OOMs the host rather than running slowly. More
        # lanes stop paying well before here anyway.
        ceiling = 4 * (os.cpu_count() or 1)
        if workers > ceiling:
            raise ValueError(
                f"n_pricing_workers={workers} exceeds {ceiling} (4x this host's "
                f"{os.cpu_count()} cores); each worker holds its own label pool"
            )
        object.__setattr__(self, "n_pricing_workers", workers)

        if isinstance(self.pricing_chunksize, bool):
            raise TypeError("pricing_chunksize must be an integer")
        try:
            chunksize = operator.index(self.pricing_chunksize)
        except TypeError as exc:
            raise TypeError("pricing_chunksize must be an integer") from exc
        if chunksize < 1:
            raise ValueError("pricing_chunksize must be positive")
        object.__setattr__(self, "pricing_chunksize", chunksize)

        if isinstance(self.lns_destroy_flights, bool):
            raise TypeError("lns_destroy_flights must be an integer")
        try:
            destroy = operator.index(self.lns_destroy_flights)
        except TypeError as exc:
            raise TypeError("lns_destroy_flights must be an integer") from exc
        if destroy < 0:
            raise ValueError("lns_destroy_flights must be non-negative")
        object.__setattr__(self, "lns_destroy_flights", destroy)
        joint_budget = float(self.lns_joint_time_limit_s)
        if not math.isfinite(joint_budget) or joint_budget < 0:
            raise ValueError("lns_joint_time_limit_s must be finite and non-negative")
        object.__setattr__(self, "lns_joint_time_limit_s", joint_budget)

        if isinstance(self.bootstrap_roots, bool):
            raise TypeError("bootstrap_roots must be an integer")
        try:
            bootstrap = operator.index(self.bootstrap_roots)
        except TypeError as exc:
            raise TypeError("bootstrap_roots must be an integer") from exc
        if not isinstance(self.bootstrap_ranking, str):
            raise TypeError("bootstrap_ranking must be a string")
        _ranking = self.bootstrap_ranking.lower()
        if _ranking not in {"score", "bound"}:
            raise ValueError("bootstrap_ranking must be 'score' or 'bound'")
        object.__setattr__(self, "bootstrap_ranking", _ranking)

        if bootstrap < 0:
            raise ValueError("bootstrap_roots must be non-negative")
        object.__setattr__(self, "bootstrap_roots", bootstrap)

        if isinstance(self.greedy_budget_s_per_flight, bool):
            raise TypeError("greedy_budget_s_per_flight must be a real number")
        try:
            per_flight = float(self.greedy_budget_s_per_flight)
        except (TypeError, ValueError) as exc:
            raise TypeError("greedy_budget_s_per_flight must be a real number") from exc
        if not math.isfinite(per_flight) or per_flight < 0.0:
            raise ValueError("greedy_budget_s_per_flight must be finite and non-negative")
        object.__setattr__(self, "greedy_budget_s_per_flight", per_flight)

        for name in ("max_iterations", "n_heuristic_tries"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            if normalized < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, normalized)

        for name in ("time_limit_s", "lp_gap", "ip_gap", "M", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real number") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)

        if self.time_limit_s <= 0.0:
            raise ValueError("time_limit_s must be positive")
        if self.ip_reserve_s is not None:
            if isinstance(self.ip_reserve_s, bool):
                raise TypeError("ip_reserve_s must be a real number or None")
            try:
                reserve = float(self.ip_reserve_s)
            except (TypeError, ValueError) as exc:
                raise TypeError("ip_reserve_s must be a real number or None") from exc
            if not math.isfinite(reserve) or not 0 <= reserve < self.time_limit_s:
                raise ValueError("ip_reserve_s must be finite and in [0, time_limit_s)")
            object.__setattr__(self, "ip_reserve_s", reserve)
        if not 0.0 <= self.lp_gap < 1.0:
            raise ValueError("lp_gap must be in [0, 1)")
        if not 0.0 <= self.ip_gap < 1.0:
            raise ValueError("ip_gap must be in [0, 1)")
        if self.M <= 0.0:
            raise ValueError("M must be positive")
        if not 0.0 <= self.epsilon < 0.5:
            raise ValueError("epsilon must be in [0, 0.5)")
