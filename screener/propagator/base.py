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
        julian_date_start: float,
        julian_date_end: float,
        time_mesh: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate a batch of TLEs to a given time and return a tensor of shape (batch_size, 6)
        with the propagated state vectors.
        """
        raise NotImplementedError
