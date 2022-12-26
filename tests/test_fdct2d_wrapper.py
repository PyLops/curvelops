import numpy as np
import pytest

import curvelops.fdct2d_wrapper as ct

pars = [
    {"nx": 100, "ny": 50, "imag": 0, "dtype": "float64"},
    {"nx": 100, "ny": 50, "imag": 1j, "dtype": "float64"},
    {"nx": 256, "ny": 256, "imag": 0, "dtype": "float64"},
    {"nx": 256, "ny": 256, "imag": 1j, "dtype": "float64"},
    {"nx": 512, "ny": 256, "imag": 0, "dtype": "float64"},
    {"nx": 512, "ny": 256, "imag": 1j, "dtype": "float64"},
    {"nx": 512, "ny": 512, "imag": 0, "dtype": "float64"},
    {"nx": 512, "ny": 512, "imag": 1j, "dtype": "complex128"},
]


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_wrapper_2dsignal(par):
    x = (
        np.random.normal(0, 1, (par["nx"], par["ny"]))
        + np.random.normal(0, 1, (par["nx"], par["ny"])) * par["imag"]
    )

    for nbscales in [4, 6, 8, 16]:
        for nbangles_coarse in [8, 16]:
            for ac in [True, False]:
                c = ct.fdct2d_forward_wrap(nbscales, nbangles_coarse, ac, x)
                xinv = ct.fdct2d_inverse_wrap(
                    *x.shape, nbscales, nbangles_coarse, ac, c
                )
                np.testing.assert_array_almost_equal(x, xinv, decimal=12)
                np.testing.assert_array_almost_equal(
                    2.0 * np.sum(np.abs(x - xinv)) / np.sum(np.abs(x + xinv)),
                    0.0,
                    decimal=12,
                )
