# A bounded root approach-and-arrival upper

The optional bound enumerates arrival times and the last graph edge, not full
routes. It runs after the initial preliminary search has supplied an incumbent,
and can retry with the remaining work budget after a future-informed search. It
adds one immutable upper per departure/lane root. Ordinary GPU expansion,
canonical completion validation and all other bounds remain in use.

For a root with lattice start time \(s\), consider every arrival
\(t\in[s+d_{\rm reverse},\min(s+H,T)]\), every destination cell \(v\), and
every prepared incoming edge \(u\to v\). Keep an edge whenever the root-to-\(u\)
hex distance plus one is at most \(t-s\). All legal complete routes occur in
this set: reverse distance and hex distance are lower bounds, while the
incoming-edge enumeration includes all roles. Ignoring exclusions, revisit
rules and role restrictions enlarges the set. If a zero-hop arrival is in the
domain, include its root/destination union without a penultimate visit.

For each alternative form the exact resource-row union of the root claims,
the penultimate visit at \(t-1\), the destination visit at \(t\), and the
canonical mandatory arrival claims at \(t\). This union is a subset of every
corresponding route's claims. Sum only its positive prices, rounded downward
at every addition; include the existing downward credit for **all** negative
snapshot prices exactly once. This yields a lower bound on the complete
route's real signed resource sum, even for overlapping origin/arrival/visit
windows. There is no selected-row truncation or paid-mask approximation.

Add the existing canonical cost lower bound for exactly \(t-s\) lattice hops,
using downward additions. The existing `CompletionBounds._roundoff_upper`
then encloses original FP64 canonical cost, `fsum` and final reduced-cost
subtractions. Geometry allowances remain unchanged. Take the **maximum** of
these RC uppers across all alternatives: a route chooses one alternative.
Take the **minimum** of this result and the independent existing RC bounds.
Never add this bound to a future or endpoint upper.

Root construction is work-bounded. Missing data, nonfinite arithmetic,
overflow, an empty enumerated set, or an incomplete work-budget scan produces
infinity, never a certificate. A candidate already above the incumbent cutoff
may terminate preparation early with infinity. Deadlines still interrupt the
pricing call. Only a completed finite root upper may close that root; the
discarded upper is retained in `closed_upper` under the existing 1e-6 rule.
The integer mode skips helpers when its existing checked root upper already
closes, then outward-encodes the prepared original-FP64 root upper and
takes its minimum with the device-computed original family in the Metal/CUDA
root kernel. An unrepresentable integer interval is ignored. Descendant
kernels need no new state or buffers; the bound is intentionally root-only.

Validation must independently enumerate canonical completions by root,
including signed prices, overlapping claim windows, alternate destinations,
legal roles, revisits, exclusions, extreme/tiny values and snapshot changes.
Forced budget/deadline failure must never publish a partial upper. Actual
Metal tests must retain full CPU-pricer/per-prune-verifier guards and audit
new root deletions against the strengthened FP64 reference. CUDA validation
is source/interface only; no NVIDIA hardware is available.

Flight 2224 diagnostic motivation: at the actual subset-solver round-two
prices, the endpoint upper is 5.305845430572 while the incumbent is
4.534625218508. The best route's extra approach charge is 0.771220196737.
Joining approach and arrival rows without reachability remains too weak;
four cheaper final approaches cannot arrive in 119 hops. Adding that lower
distance constraint gives a diagnostic upper 4.534625231176. This observation
motivates the general bound; it is not an oracle route seed or a certificate
from a capped run.
