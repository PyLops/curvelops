import numpy as np
import pytest
from pylops.utils import dottest

from curvelops import FDCT2D, FDCT3D

PYCT = False
try:
    import pyct as ct

    PYCT = True
    print(
        """
    Imported `pyct`
    """
    )

except ImportError:
    print(
        """
    Could not import `pyct` (PyCurvelab), will proceed without
    checking if both libraries match
    """
    )

pars = [
    # {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 0, 'dtype': 'float64'},
    {"nx": 32, "ny": 32, "nz": 32, "imag": 1j, "dtype": "complex128"},
    # {'nx': 32, 'ny': 32, 'nz': 64, 'imag': 0, 'dtype': 'float64'},
    {"nx": 32, "ny": 32, "nz": 64, "imag": 1j, "dtype": "complex128"},
    # {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 0, 'dtype': 'complex128'},
    {"nx": 100, "ny": 50, "nz": 20, "imag": 1j, "dtype": "complex128"},
]


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_2dsignal(par):
    """
    Tests for FDCT2D operator for 2d signal.
    """
    x = (
        np.random.normal(0.0, 1.0, (par["nx"], par["ny"]))
        + np.random.normal(0.0, 1.0, (par["nx"], par["ny"])) * par["imag"]
    )

    FDCTop = FDCT2D(dims=(par["nx"], par["ny"]), dtype=par["dtype"])

    assert dottest(
        FDCTop, *FDCTop.shape, rtol=1e-12, complexflag=0 if par["imag"] == 0 else 3
    )

    y = FDCTop * x.ravel()
    xinv = FDCTop.H * y
    np.testing.assert_array_almost_equal(xinv.reshape(*x.shape), x, decimal=14)

    if PYCT:
        FDCTct = ct.fdct2(
            x.shape,
            FDCTop.nbscales,
            FDCTop.nbangles_coarse,
            FDCTop.allcurvelets,
            cpx=False if par["imag"] == 0 else True,
        )
        y_ct = np.array(FDCTct.fwd(x)).ravel()

        np.testing.assert_array_almost_equal(y, y_ct, decimal=64)
        assert y.dtype == y_ct.dtype


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_3dsignal(par):
    """
    Tests for FDCT2D operator for 3d signal.
    """
    x = (
        np.random.normal(0.0, 1.0, (par["nx"], par["ny"], par["nz"]))
        + np.random.normal(0.0, 1.0, (par["nx"], par["ny"], par["nz"])) * par["imag"]
    )
    axes = [0, -1]
    FDCTop = FDCT2D(
        dims=(par["nx"], par["ny"], par["nz"]), axes=axes, dtype=par["dtype"]
    )

    assert dottest(
        FDCTop, *FDCTop.shape, rtol=1e-12, complexflag=0 if par["imag"] == 0 else 3
    )

    y = FDCTop * x.ravel()
    xinv = FDCTop.H * y
    np.testing.assert_array_almost_equal(xinv.reshape(*x.shape), x, decimal=14)


@pytest.mark.parametrize("par", pars)
def test_FDCT3D_3dsignal(par):
    """
    Tests for FDCT3D operator for 3d signal.
    """
    x = (
        np.random.normal(0.0, 1.0, (par["nx"], par["ny"], par["nz"]))
        + np.random.normal(0.0, 1.0, (par["nx"], par["ny"], par["nz"])) * par["imag"]
    )

    FDCTop = FDCT3D(dims=(par["nx"], par["ny"], par["nz"]), dtype=par["dtype"])

    assert dottest(
        FDCTop, *FDCTop.shape, rtol=1e-12, complexflag=0 if par["imag"] == 0 else 3
    )

    y = FDCTop * x.ravel()
    xinv = FDCTop.H * y
    np.testing.assert_array_almost_equal(xinv.reshape(*x.shape), x, decimal=14)

    if PYCT:
        FDCTct = ct.fdct3(
            x.shape,
            FDCTop.nbscales,
            FDCTop.nbangles_coarse,
            FDCTop.allcurvelets,
            cpx=False if par["imag"] == 0 else True,
        )

        y_ct = np.array(FDCTct.fwd(x)).ravel()

        np.testing.assert_array_almost_equal(y, y_ct, decimal=64)
        assert y.dtype == y_ct.dtype


@pytest.mark.parametrize("par", pars)
def test_FDCT3D_4dsignal(par):
    """
    Tests for FDCT3D operator for 4d signal.
    """
    x = (
        np.random.normal(0.0, 1.0, (par["nx"], 4, par["ny"], par["nz"]))
        + np.random.normal(0.0, 1.0, (par["nx"], 4, par["ny"], par["nz"])) * par["imag"]
    )
    axes = [0, -2, -1]
    FDCTop = FDCT3D(
        dims=(par["nx"], 4, par["ny"], par["nz"]),
        axes=axes,
        dtype=par["dtype"],
    )

    assert dottest(
        FDCTop, *FDCTop.shape, rtol=1e-12, complexflag=0 if par["imag"] == 0 else 3
    )

    x = x.ravel()
    y = FDCTop * x
    xinv = FDCTop.H * y
    np.testing.assert_array_almost_equal(xinv, x, decimal=14)
