# Current column-generation formulation and seeding

Checked against the code and the outbound-only 1,200-second FAA benchmark on
2026-09-09. This note describes the optimization model separately from the
algorithm used to solve it. The current algorithm is CG + IP every round.

## 1. Variables, routes, and objective

Let \(F\) be the outbound flight requests; \(|F|=1537\) in the benchmark.
Let \(\Delta=4\) seconds and \(b_f=\lceil t_f^{\mathrm{requested}}/\Delta\rceil\).
A column \(p\) specifies the flight, integer departure \(d_p\), endpoint lane
choices, one cruise level, and a complete ordered hex-cell path.

Let \(\mathcal P_f\) be the full route universe for this solve: the routes
the pricing oracle searches, together with the initial seed columns.
Let \(\mathcal P_f^k\subseteq\mathcal P_f\) be the columns actually in the
master at a particular point in round \(k\). This distinction matters:
the native IP optimizes over the current pool, not ungenerated trajectories.

The reported ground delay and column cost are

\[
g_{fp}=\Delta(d_p-b_f),\qquad
c_{fp}=w_g g_{fp}+w_a\left(h_{fp}+
\frac{\max(0,L_{fp}-L_f^{\mathrm{ref}})}{v}\right).
\]

Here \(L_{fp}\) is the translated enroute flown distance, and
\(L_f^{\mathrm{ref}}\) is the code's endpoint-aware enroute reference distance.
It is not the hop-count reference used to limit pricing.
Currently \(w_g=1,\ w_a=3,\ v=30\) m/s and \(h_{fp}=0\):
CG generates lateral movement without airborne hover or cruise-level changes.
Thus

\[
c_{fp}=4(d_p-b_f)+0.1\max(0,L_{fp}-L_f^{\mathrm{ref}}).
\]

The sub-step rounding from the requested time to \(b_f\Delta\) is not included
in the reported ground-delay cost. Mandatory climb/descent cost is also
excluded from this congestion objective. For the all-accepted, single-level
benchmark it is a common additive constant. The internal Column.delay_s field
stores this weighted cost when objective=total_cost, despite its field name.

Define binary \(x_{fp}\) to select route \(p\), and binary \(z_f\) to deny flight
\(f\). Let \(a_{rfp}\in\{0,1\}\) indicate whether that route claims resource row
\(r\). Let \(B_r\) be row capacity and \(l_r^{\mathrm{fixed}}\) committed occupancy.
The equivalent minimization model is

\[
\begin{aligned}
\min_{x,z}\quad&
 \sum_{f\in F}\sum_{p\in\mathcal P_f}c_{fp}x_{fp}
 +M\sum_{f\in F}z_f\\
\text{s.t.}\quad&
 \sum_{p\in\mathcal P_f}x_{fp}+z_f=1 && f\in F,\\
&
 l_r^{\mathrm{fixed}}+
 \sum_{f\in F}\sum_{p\in\mathcal P_f}a_{rfp}x_{fp}\le B_r
 &&r\in\mathcal R,\\
&x_{fp},z_f\in\{0,1\}.
\end{aligned}
\]

The code eliminates \(z_f\) and maximizes

\[
\max_x\sum_{f,p}(M-c_{fp})x_{fp},
\qquad \sum_p x_{fp}\le1,
\]

with the same capacity constraints. The two objectives sum to \(|F|M\).
Here \(M=10000\). This is a finite denial penalty, not a hard requirement
to accept every flight or a formal lexicographic objective.
All 1,537 flights were accepted in both recent runs.
Committed occupancy is zero in these batch benchmarks. There are no return
flights, aircraft-reuse equations, or outbound/return precedence equations
in this particular problem.

## 2. What the capacity rows and route domain mean

Rows have two families:

- Cell/level/time: \(r=(q,\ell,\tau)\), \(B_r=1\).
- Terminal/time: \(r=(h,\tau)\), \(B_r=\text{number of pads at terminal }h\).

For a path \(q_0,\ldots,q_H\), the visit clock is

\[
v_j=d_p+n_{\mathrm{climb}}+n_{\mathrm{origin\ lane}}+j.
\]

The current geometry derives the inclusive cell-window offsets \((-2,1)\).
Each visit contributes

\[
\{(q_j,\ell,\tau):\tau\in\{v_j-2,v_j-1,v_j,v_j+1\}\}.
\]

Customer endpoint cylinders additionally claim nearby cell/time rows across
the flight levels, including their dwell and conservative geometric/time
cover. Hub endpoints claim terminal pad rows for every grid period
overlapping their half-open dwell interval:

\[
[\,\tau\Delta,(\tau+1)\Delta\,)\cap[t_0,t_1)\ne\varnothing.
\]

The column's claim set \(\mathcal C_{fp}\) is the UNION of these contributions:
\(a_{rfp}=\mathbf1\{r\in\mathcal C_{fp}\}\). A flight claiming one row in more
than one way still contributes coefficient one. Terminal pad rows are shared
by takeoff and arrival operations. Permanent terminal-airspace cylinders are
handled by route exclusion and exact per-column geometric certification.

The pricing route domain uses:

- Departure \(b_f\le d_p\le b_f+900\), i.e. at most 3,600 seconds reported ground delay.
- The single 100 m cruise level and the prescribed endpoint lane geometry.
- Adjacent lateral hex hops, at least one hop, and no hover/vertical cruise moves.
- Hop bound \(H\le D_f^{\mathrm{hex}}+3\), where the reference is hex distance
  between the snapped request origin/destination cells.
- The corresponding spatial corridor, with endpoint lanes admitted and static
  terminal exclusions applied.
- No revisit within the last three visited cells for the current four-period
  claim window.
- Canonical trajectory, wall, and detour validation; the benchmark's independent
  translated enroute-distance ratio limit is 100.

The per-flight final air-state bound is derived as

\[
T_f^{\max}=b_f+900+n_{\mathrm{climb}}
+\max_{\text{origin lanes}}n_{\mathrm{lane}}+(D_f^{\mathrm{hex}}+3).
\]

It is not a separate constraint requiring arrival before the 1,200-second
demand-generation window. That window only determines which requests are
generated. These route restrictions define the modeled search space;
certified LP bounds refer to that space.

## 3. LP, pricing, IP, and stopping

The restricted LP replaces \(\mathcal P_f\) by \(\mathcal P_f^k\) and relaxes
integrality. Let \(\pi_f\ge0\) be the dual of the flight row and
\(\lambda_r\ge0\) the dual of a capacity row in the maximize formulation.
Pricing computes one best reduced cost per flight:

\[
\rho_f^k=\max_{p\in\mathcal P_f}
\left[M-c_{fp}-\pi_f^k-\sum_r\lambda_r^k a_{rfp}\right].
\]

A positive reduced cost, subject to the numerical gate, offers an improving
column. The oracle ordinarily returns at most one new column per flight per
sweep. Known columns are included in pricing's incumbent accounting.
Capacity rows are separated until the LP solution is feasible for all its
implicit capacity constraints; absent row duals are zero.

After adding the sweep's columns, the IP solves the same master with binary
variables over the enlarged pool, warm-started from the best known feasible
schedule. The benchmark uses a 30-second native search cap per round, eager
capacity-row preparation, and a final native IP cap of 600 seconds.

If \(Z_{\mathrm{LP}}^k\) is restricted-LP revenue, exact pricing yields the
full-LP revenue upper bound

\[
U^k=Z_{\mathrm{LP}}^k+\sum_f\max(0,\rho_f^k).
\]

The code retains the best valid bound \(U_{\mathrm{best}}\), up to numerical
consistency checks. Transforming to minimization gives

\[
\underline C=|F|M-U_{\mathrm{best}},\quad
C_{\mathrm{RLP}}^k=|F|M-Z_{\mathrm{LP}}^k,\quad
\frac{\max(0,C_{\mathrm{RLP}}^k-\underline C)}
{\max(1,|C_{\mathrm{RLP}}^k|)}\le0.001.
\]

This is the 0.1% LP stopping condition. \(C_{\mathrm{RLP}}\) is an upper bound
on the full LP optimum, not on the integer optimum. For a feasible integer
schedule cost \(C_{\mathrm{IP}}\), the corresponding global integer gap is
\((C_{\mathrm{IP}}-\underline C)/\max(1,|C_{\mathrm{IP}}|)\).
Proving pool-IP optimality does not by itself prove global route optimality.

The optional cheap mode restricts pricing to a promising departure/lane root
and retains any better known candidate. Its scores cannot certify the maximum
over the full universe, so only exact sweeps update the global bound and permit
the LP-target stop. It changes the solution algorithm, not the master objective.

## 4. Seeding, explicitly

Seeding has three different jobs.

First, construct one nominal-departure spatial seed independently per flight.
Let \(\mathcal B_f\) contain the canonically valid deterministic shortest-path
candidates constructed for endpoint lane pairs at \(d=b_f\). The usual path is

\[
p_f^0\in\arg\min_{p\in\mathcal B_f}c_{fp}.
\]

The implementation orders lane pairs by admissible bounds, builds deterministic
BFS paths, and chooses the best valid candidate with deterministic tie-breaking.
There are no congestion dual prices and no joint capacity constraints between
flights in this step. If the fast candidate construction finds no valid seed,
the code attempts a bounded zero-dual DAG fallback. That fallback is a feasible
seed, not a universal minimum-cost certificate.

Second, define the pure time-translation operator

\[
\mathsf T_k p=(d_p+k,\text{same level, lanes and spatial path}),\quad k\ge0.
\]

It obeys

\[
c_f(\mathsf T_kp)=c_f(p)+w_gk\Delta,\qquad
\mathcal C_f(\mathsf T_kp)=
\{(\text{same resource},\tau+k):(\text{resource},\tau)\in\mathcal C_f(p)\}.
\]

Initialize the pool with the legal ladder

\[
\mathcal L_f=\{\mathsf T_kp_f^0:k=0,\ldots,20\}.
\]

Thus every flight gets the same route at 0, 4, 8, ..., 80 additional seconds,
truncated if its legal departure/path-clock limits require it. These are
different timed columns with the SAME spatial route. They are alternatives
for the LP/IP, not 21 flights being executed and not 21 route geometries.

Third, build a feasible warm start greedily. Order flights by requested
departure time, breaking ties by flight ID. Start with
\(\ell_r=l_r^{\mathrm{fixed}}\). For each flight choose

\[
k_f^\star=\min\left\{k\in\mathbb Z_{\ge0}:
\mathsf T_kp_f^0\text{ is legal},\
\ell_r+a_{r,f,\mathsf T_kp_f^0}\le B_r\quad\forall r
\right\}.
\]

When a legal shift exists, set \(\widehat p_f=\mathsf T_{k_f^\star}p_f^0\)
and update \(\ell_r\leftarrow\ell_r+a_{rf\widehat p_f}\).
This pass can search beyond the 20-rung ladder, up to the actual flight
delay/path-clock limits. If it cannot fit a flight, the initial selection can
be partial. In this benchmark it fits all flights.

The initial per-flight pool and warm start are then

\[
\mathcal P_f^0=\mathcal L_f\cup\{\widehat p_f\},
\qquad \widehat x_{f,\widehat p_f}=1.
\]

Deduplicating equal columns, this benchmark has
\(1537+20(1537)+144=32421\) initial columns. The greedy starting schedule costs
199,759.69245 congestion units. It supplies an integer feasible solution and
thus a cost upper bound. The LP can mix fractions of any seed alternatives;
subsequent IPs can replace the warm start with any feasible combination in
the growing pool. Neither a seed's route nor its greedy departure is fixed.

## Source map

- [Objective and weights](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/objective.py:31)
- [Master column coefficients](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/master.py:792)
- [Resource claims](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/network.py:1627)
- [Nominal seed construction](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/pricing.py:3409)
- [Time shifts, ladder, and greedy schedule](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/solver.py:146)
- [Benchmark inputs](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_cheap_pricing_1200s_20260909/density_faa_wing_zipline_seed0_iteration_ip_eager/inputs.json)

## Illustrated coefficient derivation

The follow-up [ledger and claim comparison](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_claims_20260909/README.md)
contains spatial and temporal PNGs built from the actual A*/CG reservation
builders and the A* occupancy service. It separates identical filed geometry,
CG source resource rows, and A* candidate-entry exclusions on a controlled
two-hop excerpt. The derivation includes the extra conservative boundary step
in A*'s current occupancy raster.
