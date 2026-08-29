import jax
import jax.numpy as jnp
import torch.utils.dlpack
from jaxsgp4 import tle2sat_array, sgp4_jdfr
import torch

from .base import BasePropagator, RVPairs

jax.config.update("jax_enable_x64", True)


class JAXSGP4Propagator(BasePropagator):

    def propagate(
        self,
        tle_batch: list[tuple[str, str]],
        julian_date_start: tuple[float, float],
        julian_date_end: tuple[float, float],
        time_mesh: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate a batch of TLEs to a given time and return a tensor of shape (batch_size, 6)
        with the propagated state vectors.
        """
        assert julian_date_start[0] == julian_date_end[0], (
            "For convenience, we only support the same integer part of Julian date for now, "
            "while the fractional part can be out of [0,1].")

        # Convert TLEs to batched Satellite objects
        tle_1_lines, tle_2_lines = zip(*tle_batch)
        sats = tle2sat_array(tle_1_lines, tle_2_lines)

        propagate_times = jax.vmap(
            sgp4_jdfr,
            in_axes=(None, 0, 0),
        )

        propagate_all = jax.vmap(
            propagate_times,
            in_axes=(0, None, None),
        )

        propagate_all = jax.jit(propagate_all)

        frs = jnp.arange(
            julian_date_start[1],
            julian_date_end[1] + time_mesh / 86400.0,
            time_mesh / 86400.0,
        )
        jds = jnp.full_like(frs, julian_date_start[0])

        # Propagate all satellites
        rvs, errors = propagate_all(sats, jds, frs)
        rs = rvs[..., :3]
        vs = rvs[..., 3:]

        rs = torch.utils.dlpack.from_dlpack(rs)
        vs = torch.utils.dlpack.from_dlpack(vs)

        return rs, vs
