"""Plan a round trip as ONE flight: ``origin -> dest -> (service) -> origin``.

The two legs of a delivery are flown by the same aircraft, so the return cannot leave before the
outbound has arrived. Filing them as two independent requests cannot express that: the return's
departure has to be guessed at demand-generation time, from a straight-line, empty-sky estimate that
knows nothing about ground hold, lattice overhead or detour. Measured on density_faa, that estimate
was short on every one of 2,318 round trips — 85 were filed to depart before their aircraft could
possibly land, and the median surviving pair had 9.3 s of slack.

Here the return leg's departure is not estimated. It is read off the outbound leg that was actually
planned, so precedence is a property of the construction rather than something a checker has to look
for afterwards.

**Chained, not joint.** Leg 2 is planned after leg 1, against leg 1's outcome — it is not a single
search over both legs. A joint search would need a leg dimension in the A* state, which the compiled
kernel, the fixed-lane takeoff/landing gates and the heuristic's admissibility corrections all key
off; the cost is that leg 1 is chosen without knowing what it does to leg 2. That is exactly the
trade the two-request scheme already made, so nothing regresses — what changes is that the legs are
now *sequentially consistent* instead of independently timed.
"""
from __future__ import annotations

from dataclasses import replace

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import DenialReason, FlightRequest, IntentStatus, OperationalIntent
from ..volumes import ground_dwell_reservation


class ItineraryPlanner:
    """Wraps a leg planner and composes a round trip into a single :class:`OperationalIntent`.

    A request without ``return_to_origin`` is forwarded untouched, so this is transparent to every
    one-way flight and can wrap any planner in the chain.
    """

    def __init__(self, inner):
        self.inner = inner

    # --- the wrapper-chain contract the rest of the codebase duck-types -------------------------
    def __getattr__(self, name):
        # Markers and optional members (`plans_whole_schedule`, `capacity_authority`,
        # `record_envelope`, `last_envelope`, ...) belong to the planner that actually plans. Only
        # reached for attributes this class does not define, so `inner` itself never recurses.
        return getattr(self.inner, name)

    def plan(self, req: FlightRequest, ledger: ReservationLedger,
             cfg: SimConfig) -> OperationalIntent:
        if not req.return_to_origin:
            return self.inner.plan(req, ledger, cfg)

        # --- leg 1: origin -> dest ---------------------------------------------------------------
        out = self.inner.plan(self._leg(req, req.origin, req.dest, req.origin_terminal,
                                        req.dest_terminal, req.t_departure), ledger, cfg)
        if not out.accepted:
            return replace(out, request=req)

        # The aircraft is DOWN when its arrival column ends: `realized_release_s` is that column's
        # `t_end`, i.e. the descent is already flown. The service sits on top of it, and the return
        # then takes off — so the return departs after the arrival that ACTUALLY happened, whatever
        # hold or detour leg 1 took.
        from ..sim import realized_release_s          # sim imports planners; keep one owner
        landed = realized_release_s(out)
        if landed is None:                            # accepted with no volumes: nothing to chain
            return replace(out, request=req)
        depart_back = float(landed) + float(req.service_time_s)

        # --- leg 2: dest -> origin ---------------------------------------------------------------
        # NOT deconflicted against leg 1: both legs are the same aircraft, and an aircraft does not
        # conflict with itself. Under the two-request scheme they were separate flight_ids and the
        # ledger DID police them against each other, which was never physically meaningful.
        back = self.inner.plan(self._leg(req, req.dest, req.origin, req.dest_terminal,
                                         req.origin_terminal, depart_back), ledger, cfg)
        if not back.accepted:
            # The itinerary is atomic: a delivery whose aircraft cannot get home is not a half
            # success. Denying the whole thing keeps `accepted` meaning "this aircraft flew the
            # trip", so nothing downstream has to special-case a stranded leg.
            return OperationalIntent(
                request=req, status=IntentStatus.REJECTED, volumes=[], centerline=[],
                denial_reason=back.denial_reason, planner=back.planner,
                solve_time_s=out.solve_time_s + back.solve_time_s)

        return self._compose(req, out, back, float(landed), cfg)

    @staticmethod
    def _leg(req, origin, dest, o_term, d_term, t_departure) -> FlightRequest:
        """One leg as an ordinary one-way request. ``t_request`` is pulled down to the leg's own
        departure because a return leg departs long after the itinerary was filed, and
        ``FlightRequest`` forbids departing before filing."""
        return replace(req, origin=origin, dest=dest, origin_terminal=o_term, dest_terminal=d_term,
                       t_request=min(req.t_request, t_departure), t_departure=t_departure,
                       return_to_origin=False, service_time_s=0.0)

    @staticmethod
    def _compose(req, out, back, landed, cfg) -> OperationalIntent:
        """One intent holding both legs and the pad between them.

        The dwell is a LOW box, not the full column (``volumes.ground_dwell_reservation``): the
        aircraft is on the ground, while the descent before it and the climb after it each sweep the
        whole column and are already reserved as leg 1's arrival and leg 2's departure columns.

        It spans arrival column END -> departure column START, not merely ``service_time_s``. Two
        things open a gap between those: the return's departure snaps up to the ``dt`` grid, and a
        congested return may be ground-held well past its service. The aircraft is parked for every
        second of that, so anything short would leave the pad reservable underneath a parked drone.
        """
        from ..verify import realized_takeoff_s
        dwell = []
        leaves = realized_takeoff_s(back)
        if leaves is not None and leaves > landed + 1e-9:
            from ..volumes import terminal_radius
            d_term = req.dest_terminal
            dwell = [ground_dwell_reservation(
                req.dest, landed, float(leaves) - landed, cfg,
                terminal_id=d_term.id if d_term is not None else None,
                radius=terminal_radius(d_term, cfg) if d_term is not None else None)]
        return OperationalIntent(
            request=req,
            status=out.status,
            volumes=[*(out.volumes or []), *dwell, *(back.volumes or [])],
            centerline=[*(out.centerline or []), *(back.centerline or [])],
            leg_starts=(len(out.centerline or []),),
            ground_delay_s=out.ground_delay_s + back.ground_delay_s,
            air_hold_s=out.air_hold_s + back.air_hold_s,
            air_detour_m=out.air_detour_m + back.air_detour_m,
            lattice_overhead_m=out.lattice_overhead_m + back.lattice_overhead_m,
            altitude_change_m=out.altitude_change_m + back.altitude_change_m,
            cost=out.cost + back.cost,
            denial_reason=DenialReason.NONE,
            planner=out.planner,
            solve_time_s=out.solve_time_s + back.solve_time_s,
        )
