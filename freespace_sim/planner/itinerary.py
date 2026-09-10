"""Round-trip itineraries: one flight, two legs, one reservation.

See `context/figures/itinerary_reservation.png` for the reservation the composed intent holds.
"""
from __future__ import annotations

from dataclasses import replace

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import DenialReason, FlightRequest, IntentStatus, OperationalIntent
from ..volumes import ground_dwell_reservation, terminal_radius
from . import iter_planner_chain


def reject_itinerary(req: FlightRequest, planner: str) -> None:
    """
    Refuse a round-trip request on a planner that can only plan one leg.

    A single-leg planner handed a `return_to_origin` request plans the outbound and drops the return
    without a trace — and because that costs about half a round trip, a cost-comparing caller reads
    the loss as an improvement. Fail loudly instead; wrap the planner in `ItineraryPlanner`.

    Parameters
    ------------
    - req (FlightRequest): the request about to be planned
    - planner (str): name used in the error, so the offending construction site is identifiable

    Return
    --------
    - None; raises NotImplementedError when `req.return_to_origin` is set.
    """
    if req.return_to_origin:
        raise NotImplementedError(
            f"{planner} cannot plan a round-trip itinerary (flight {req.flight_id}): it plans one "
            "origin->dest leg, so the return would be dropped without a trace. Wrap it in "
            "ItineraryPlanner (planner.get_planner does this) or file the legs separately.")


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
        """Read markers and optional members off the planner that plans.

        `inner` and `warm_planner` are refused rather than forwarded. `inner` because this hook runs
        on any instance whose dict is empty — the shell `copy`/`pickle` build before calling
        `__setstate__` — where forwarding recurses until the stack dies. `warm_planner` because this
        wrapper has none of its own: answering with the inner planner's would make
        `iter_planner_chain` yield a grandchild at this wrapper's depth, and that order is
        load-bearing (`_terminal_capacity_for` takes the FIRST match).
        """
        if name in ("inner", "warm_planner"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        """Write configuration through to the inner planner, so a set and a get agree.

        `evict_floor`, `record_envelope` and the kernel knobs are set on whatever object a caller
        holds. Storing them here instead would leave the getter reporting the new value while the
        planner that actually plans kept the old one.
        """
        if name == "inner" or name in type(self).__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

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
            raise ValueError(
                f"flight {req.flight_id}: the outbound leg was ACCEPTED with no volumes, so there is "
                "no arrival for the return to depart from. An accepted intent must carry the volumes "
                "it conflict-checked (see planner.Planner). Returning the outbound alone here would "
                "be the silent half-trip reject_itinerary exists to prevent.")
        dwell = cfg.turnaround_s if req.turnaround_s is None else req.turnaround_s

        # Leg 2 is NOT deconflicted against leg 1: both are the same aircraft, and an aircraft does
        # not conflict with itself.
        leg1_reads = self._recorded_envelopes()
        back = self.inner.plan(self._leg(req, req.dest, req.origin, req.dest_terminal,
                                         req.origin_terminal, float(landed) + dwell),
                               ledger, cfg)
        self._absorb_envelopes(leg1_reads)
        if not back.accepted:
            return OperationalIntent(
                request=req, status=IntentStatus.REJECTED, volumes=[], centerline=[],
                denial_reason=back.denial_reason, planner=back.planner,
                solve_time_s=out.solve_time_s + back.solve_time_s)
        return self._compose(req, out, back, float(landed), cfg, ledger)

    def _recorded_envelopes(self):
        """Every chained planner holding a read envelope right now, paired with what it holds.

        Empty unless a caller turned `record_envelope` on, which only the parallel engines do.
        """
        return [(p, p.last_envelope) for p in iter_planner_chain(self.inner)
                if getattr(p, "last_envelope", None) is not None]

    @staticmethod
    def _absorb_envelopes(earlier) -> None:
        """Fold an earlier leg's read set into the one its planner holds now.

        A planner clears `last_envelope` per `plan` call, so without this an itinerary is summarised
        by its LAST leg and `parallel.envelope_intersects` calls a commit inside the outbound's read
        set clean — the speculation is kept and the run diverges from sequential.
        """
        for planner, env in earlier:
            now = getattr(planner, "last_envelope", None)
            planner.last_envelope = env if now is None else now.union(env)

    @staticmethod
    def _leg(req, origin, dest, o_term, d_term, t_departure) -> FlightRequest:
        """One leg as an ordinary one-way request, keeping the itinerary's filing time."""
        return replace(req, origin=origin, dest=dest, origin_terminal=o_term, dest_terminal=d_term,
                       t_departure=t_departure, return_to_origin=False, turnaround_s=0.0)

    @staticmethod
    def _compose(req, out, back, landed, cfg, ledger) -> OperationalIntent:
        """Join both legs and the pad dwell into one intent.

        The dwell spans arrival-column end -> departure-column start, not merely `turnaround_s`: a
        congested return can be held past its service and the aircraft is parked for all of it, so
        anything shorter leaves the pad reservable underneath a parked drone.
        """
        from ..verify import realized_takeoff_s
        dwell = []
        leaves = realized_takeoff_s(back)
        if leaves is not None and leaves > landed + 1e-9:
            d_term = req.dest_terminal
            # NOT tagged with the terminal: `conflict.volumes_conflict` makes two same-hub volumes
            # transparent whenever either is a cylinder, and this box is one. Tagging it would let
            # another flight of the same hub land its column on top of the parked aircraft.
            dwell = [ground_dwell_reservation(
                req.dest, landed, float(leaves) - landed, cfg,
                radius=terminal_radius(d_term, cfg) if d_term is not None else None)]
            # This box is derived from both legs' results, so neither search deconflicted it; check
            # it against the ledger here, where the answer is still a denial the caller can read.
            # `FCFSMechanism.commit` re-checks and would catch it, but reports it as a lost
            # commit-time race, and the LNS commit path (`lns/state.py`) re-checks nothing at all.
            if ledger.any_conflict(dwell):
                return OperationalIntent(
                    request=req, status=IntentStatus.REJECTED, volumes=[], centerline=[],
                    denial_reason=DenialReason.CONFLICT_FILED, planner=out.planner,
                    solve_time_s=out.solve_time_s + back.solve_time_s)
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
