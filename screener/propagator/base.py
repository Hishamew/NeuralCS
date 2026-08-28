__all__ = ["BasePropagator"]

from typing import NamedTuple
import torch


class RVPairs(NamedTuple):
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


class BasePropagator:

    def propagate(
        self,
        tle_batch: list[tuple[str, str]],
        time_mesh: float,
        end_time: float,
    ) -> list[RVPairs]:
        """
        Propagate a batch of TLEs to a given time and return a tensor of shape (batch_size, 6)
        with the propagated state vectors.
        """
        raise NotImplementedError
