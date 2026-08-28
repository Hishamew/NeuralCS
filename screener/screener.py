from .propagator import BasePropagator

import torch


class Screener:

    def __init__(
        self,
        coarse_mesh: float = 60,
        fine_mesh: float = 1,
        propagator: BasePropagator = None,
    ):
        self.coarse_mesh = coarse_mesh
        self.fine_mesh = fine_mesh
        self.propagator = propagator

    def screen(self, tle_batch: list[tuple[str, str]]) -> torch.Tensor:
        """
        Screen a batch of TLEs and return a tensor of shape (batch_size, batch_size) with
        the all vs all conjunction screening results.
        """

    def pre_screen(self, tle_batch: list[tuple[str, str]]) -> torch.Tensor:
        """
        Pre-screen a batch of TLEs and return a tensor of shape (batch_size, batch_size) with
        the all vs all conjunction screening results.
        """

    def coarse_screen(self, tle_batch: list[tuple[str, str]]) -> torch.Tensor:
        """
        Screen a batch of TLEs and return a tensor of shape (batch_size, batch_size) with
        the all vs all conjunction screening results.
        """

    def fine_screen(self, tle_batch: list[tuple[str, str]]) -> torch.Tensor:
        """
        Screen a batch of TLEs and return a tensor of shape (batch_size, batch_size) with
        the all vs all conjunction screening results.
        """

    def _linear_simple_conjunction_assesment(
        self,
        rvt_0: torch.Tensor,
        rvt_1: torch.Tensor,
        with_miss_distance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the linear conjunction assessment between two state vectors.
        rvt_0: tensor of shape (batch_size, 6)
        rvt_1: tensor of shape (batch_size, 6)
        return: tensor of shape (batch_size,) with the conjunction assessment results.
        """
        raise NotImplementedError
