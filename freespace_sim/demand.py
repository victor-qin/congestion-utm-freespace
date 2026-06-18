"""Demand models — how flight requests are generated.

`UniformPoissonDemand`: Poisson(λ) arrivals over the horizon, origin/dest sampled uniformly in the
region at ground level, with a minimum O/D separation so requests are non-trivial. Deterministic
under a seeded RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import SimConfig
from .types import FlightRequest, vec


class DemandModel(Protocol):
    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]: ...


@dataclass
class UniformPoissonDemand:
    min_od_separation_m: float = 1000.0
    uss_ids: tuple[str, ...] = ("default",)

    def generate(self, cfg: SimConfig, rng: np.random.Generator) -> list[FlightRequest]:
        w, h = cfg.region_size_m
        n = int(rng.poisson(cfg.lam_per_hour * cfg.horizon_s / 3600.0))
        requests: list[FlightRequest] = []
        for fid in range(n):
            for _ in range(20):  # rejection-sample until O/D are far enough apart
                o = rng.uniform([0, 0], [w, h])
                d = rng.uniform([0, 0], [w, h])
                if np.linalg.norm(d - o) >= self.min_od_separation_m:
                    break
            t_request = float(rng.uniform(0, cfg.horizon_s))
            uss_id = self.uss_ids[int(rng.integers(len(self.uss_ids)))]
            requests.append(
                FlightRequest(
                    flight_id=fid,
                    origin=vec(o[0], o[1], cfg.ground_level_m),
                    dest=vec(d[0], d[1], cfg.ground_level_m),
                    t_request=t_request,
                    uss_id=uss_id,
                )
            )
        requests.sort(key=lambda r: (r.t_request, r.flight_id))
        return requests
