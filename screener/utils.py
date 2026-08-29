import numpy as np
from numpy.typing import NDArray
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import (
    TEME,
    PrecessedGeocentric,
    CartesianRepresentation,
    CartesianDifferential,
)


def teme_to_j2000(
    r_teme: NDArray,
    v_teme: NDArray,
    time: Time,
) -> tuple[NDArray, NDArray]:
    """
    Convert TEME coordinates to J2000 coordinates.

    Parameters
    ----------
    r_teme : array-like
        Position vector in TEME frame (km).
    v_teme : array-like
        Velocity vector in TEME frame (km/s).
    time : astropy.time.Time
        Time of the observation.

    Returns
    -------
    r_j2000 : array-like
        Position vector in J2000 frame (km).
    v_j2000 : array-like
        Velocity vector in J2000 frame (km/s).
    """
    position = CartesianRepresentation(np.asarray(r_teme) * u.km)
    velocity = CartesianDifferential(np.asarray(v_teme) * u.km / u.s)
    state = position.with_differentials(velocity)

    teme = TEME(
        state,
        obstime=time,
    )

    j2000 = teme.transform_to(
        PrecessedGeocentric(equinox=Time('J2000'), obstime=time))

    r_j2000 = j2000.cartesian.xyz.to_value(u.km)
    v_j2000 = j2000.velocity.d_xyz.to_value(u.km / u.s)

    return r_j2000, v_j2000
