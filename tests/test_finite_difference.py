import pytest

import numpy as np

from sparseappbench.benchmarks.Finite_Difference import (
    buckley_leverett_flux,
    burgers_flux,
    lax_freidrichs_data_generator,
    lax_friedrichs_solver,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "dt, dx, expect_stable",
    [
        (0.01, 0.1, True),  # CFL << 1  → stable
        (0.05, 0.1, True),  # CFL ≈ 0.5 → stable
        (0.2, 0.1, False),  # CFL > 1   → unstable
    ],
)
def test_lax_friedrichs_cfl_burgers(dt, dx, expect_stable):
    xp = NumpyFramework()
    u_0 = lax_freidrichs_data_generator(xp, 200)
    timesteps = 50
    fluxes_bench = lax_friedrichs_solver(
        xp,
        u_0,
        dt=dt,
        dx=dx,
        flux=burgers_flux,
        timesteps=timesteps,
        boundary="dirichlet",
    )

    fluxes = xp.from_benchmark(fluxes_bench)

    max_flux = np.max(np.abs(fluxes))

    if expect_stable:
        assert np.isfinite(max_flux)
    else:
        assert not np.isfinite(max_flux)


def max_wave_speed_buckley(u):
    eps = 1e-6
    return np.max(
        np.abs(
            (buckley_leverett_flux(u + eps) - buckley_leverett_flux(u - eps))
            / (2 * eps)
        )
    )


# Now max for


@pytest.mark.parametrize(
    "dt, dx, expect_stable",
    [
        (0.01, 0.1, True),  # CFL << 1  → stable
        (0.05, 0.1, True),  # CFL ≈ 0.5 → stable
        (0.2, 0.1, False),  # CFL > 1   → unstable
    ],
)
def test_lax_friedrichs_cfl_buckley_leverett_flux(dt, dx, expect_stable):
    xp = NumpyFramework()
    u_0 = lax_freidrichs_data_generator(xp, 200)
    timesteps = 50
    cmax = max_wave_speed_buckley(u_0)
    fluxes_bench = lax_friedrichs_solver(
        xp,
        u_0,
        dt=dt,
        dx=dx,
        flux=buckley_leverett_flux,
        timesteps=timesteps,
        boundary="dirichlet",
    )

    fluxes = xp.from_benchmark(fluxes_bench)

    max_flux = np.max(np.abs(fluxes))

    if cmax * dt / dx <= 1:
        assert np.isfinite(max_flux)
    else:
        assert not np.isfinite(max_flux)
