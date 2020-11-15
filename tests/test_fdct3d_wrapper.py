import pytest
import curvelops.fdct3d_wrapper as ct
import numpy as np

pars = [
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 0, 'dtype': 'float64'},
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 1j, 'dtype': 'complex128'},
    {'nx': 32, 'ny': 32, 'nz': 64, 'imag': 0, 'dtype': 'float64'},
    {'nx': 32, 'ny': 32, 'nz': 64, 'imag': 1j, 'dtype': 'complex128'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 0, 'dtype': 'float64'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 1j, 'dtype': 'complex128'},
]


@pytest.mark.parametrize("par", pars)
def test_FDCT3D_wrapper_3dsignal(par):
    x = np.random.normal(0, 1, (par['nx'], par['ny'], par['nz'])) + \
        np.random.normal(0, 1, (par['nx'], par['ny'], par['nz'])) * par['imag']
    for nbscales in [4, 6, 8]:
        for nbangles_coarse in [8, 16]:
            for ac in [True, False]:
                c = ct.fdct3d_forward_wrap(nbscales, nbangles_coarse, ac, x)
                xinv = ct.fdct3d_inverse_wrap(
                    *x.shape, nbscales, nbangles_coarse, ac, c)
                np.testing.assert_array_almost_equal(x, xinv, decimal=12)
                np.testing.assert_array_almost_equal(
                    2.*np.sum(np.abs(x-xinv))/np.sum(np.abs(x+xinv)), 0., decimal=12)
