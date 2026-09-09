"""Round-trip itineraries: one flight, two legs, one reservation.

See `context/figures/itinerary_reservation.png` for the reservation the composed intent holds.
"""
from __future__ import annotations

from dataclasses import replace

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import DenialReason, FlightRequest, IntentStatus, OperationalIntent
from ..volumes import ground_dwell_reservation


class ItineraryPlanner:
    """
    Compose a `return_to_origin` request into one intent covering both legs and the pad between them.

    Leg 2 departs from the arrival leg 1 actually achieved, so a return cannot precede its own
    outbound. Chained, not joint: leg 2 is planned against leg 1's outcome rather than searched
    jointly with it, because a joint search needs a leg dimension in the A* state that the compiled
    kernel, the fixed-lane gates and the heuristic's admissibility corrections all key off.

    Parameters
    ------------
    - inner (Planner): the per-flight planner that plans each leg. A one-way request is forwarded to
      it untouched, so wrapping is transparent to every flight that is not a round trip.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        # Markers and optional members (`plans_whole_schedule`, `capacity_authority`, ...) belong to
        # the planner that plans. Only reached for names this class does not define, so `inner`
        # itself cannot recurse.
        return getattr(self.inner, name)

    def plan(self, req: FlightRequest, ledger: ReservationLedger,
             cfg: SimConfig) -> OperationalIntent:
        """
        Plan a round trip as one flight, or forward a one-way request to the inner planner.

        Parameters
        ------------
        - req (FlightRequest): the request; `return_to_origin` selects the itinerary path
        - ledger (ReservationLedger): committed traffic both legs deconflict against
        - cfg (SimConfig): simulation configuration

        Return
        --------
        - intent (OperationalIntent): both legs and the pad between them, or the inner planner's
          result unchanged. Denied whole if either leg is denied — a delivery whose aircraft cannot
          get home is not a half success.
        """
        if not req.return_to_origin:
            return self.inner.plan(req, ledger, cfg)

        out = self.inner.plan(self._leg(req, req.origin, req.dest, req.origin_terminal,
                                        req.dest_terminal, req.t_departure), ledger, cfg)
        if not out.accepted:
            return replace(out, request=req)

        from ..sim import realized_release_s          # sim imports planners; keep one owner
        landed = realized_release_s(out)
        if landed is None:
            return replace(out, request=req)

        # Leg 2 is NOT deconflicted against leg 1: both are the same aircraft, and an aircraft does
        # not conflict with itself.
        back = self.inner.plan(self._leg(req, req.dest, req.origin, req.dest_terminal,
                                         req.origin_terminal, float(landed) + req.service_time_s),
                               ledger, cfg)
        if not back.accepted:
            return OperationalIntent(
                request=req, status=IntentStatus.REJECTED, volumes=[], centerline=[],
                denial_reason=back.denial_reason, planner=back.planner,
                solve_time_s=out.solve_time_s + back.solve_time_s)
        return self._compose(req, out, back, float(landed), cfg)

    @staticmethod
    def _leg(req, origin, dest, o_term, d_term, t_departure) -> FlightRequest:
        """One leg as an ordinary one-way request.

        `t_request` is pulled down to the leg's own departure: a return leg departs long after the
        itinerary was filed, and `FlightRequest` forbids departing before filing.
        """
        return replace(req, origin=origin, dest=dest, origin_terminal=o_term, dest_terminal=d_term,
                       t_request=min(req.t_request, t_departure), t_departure=t_departure,
                       return_to_origin=False, service_time_s=0.0)

    @staticmethod
    def _compose(req, out, back, landed, cfg) -> OperationalIntent:
        """Join both legs and the pad dwell into one intent.

        The dwell spans arrival-column end -> departure-column start, not merely `service_time_s`: a
        congested return can be held past its service and the aircraft is parked for all of it, so
        anything shorter leaves the pad reservable underneath a parked drone.
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
