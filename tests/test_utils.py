import pytest
import numpy as np
from numpy.random import randint
from curvelops import FDCT
from curvelops.utils import (
    array_split_nd,
    apply_along_wedges,
    split_nd,
    energy,
    energy_split,
    ndargmax,
)


pars = [
    {"shape": (randint(1, 99),), "splits": (randint(1, 10),)},
    {
        "shape": (randint(1, 99), randint(1, 99)),
        "splits": (randint(1, 10), randint(1, 10)),
    },
    {
        "shape": (randint(1, 99), randint(1, 99), randint(1, 99)),
        "splits": (randint(1, 10), randint(1, 10), randint(1, 10)),
    },
]

pars_cl = [
    {"shape": (randint(32, 129), randint(32, 129))},
    {"shape": (randint(32, 129), randint(32, 129), randint(32, 129))},
]


def test_array_split_nd_simple():
    x = np.outer(1 + np.arange(2), 2 + np.arange(3))
    y = array_split_nd(x, 2, 3)
    assert len(x) == 2
    for subx in x:
        assert len(subx) == 3
    assert y[0][0] == 2
    assert y[0][1] == 3
    assert y[0][2] == 4
    assert y[1][0] == 4
    assert y[1][1] == 6
    assert y[1][2] == 8


@pytest.mark.parametrize("par", pars)
def test_array_split_nd_sizes(par):
    shape = par["shape"]
    splits = par["splits"]
    x = np.zeros(tuple(a * b for (a, b) in zip(shape, splits)))
    y = array_split_nd(x, *splits)
    for split in splits:
        assert split == len(y)
        y = y[0]
    assert y.shape == shape


@pytest.mark.parametrize("par", pars)
def test_split_nd_sizes(par):
    shape = par["shape"]
    splits = par["splits"]
    x = np.zeros(tuple(a * b for (a, b) in zip(shape, splits)))
    y = split_nd(x, *splits)
    for split in splits:
        assert split == len(y)
        y = y[0]
    assert y.shape == shape


@pytest.mark.parametrize("par", pars_cl)
def test_apply_along_wedges(par):
    shape = par["shape"]
    Cop = FDCT(shape, axes=list(range(len(shape))))
    x = (
        np.random.normal(0.0, 1.0, shape)
        + np.random.normal(0.0, 1.0, shape) * 1j
    )
    # Create a vector of curvelet coeffs
    y = Cop @ x
    # Convert to structure
    y_struct = Cop.struct(Cop @ x)
    # Add 1 to each wedge
    y_struct_one = apply_along_wedges(
        y_struct,
        lambda c, w, s, na, ns: c + 1.0,
    )
    # Convert back to vector
    y_one = Cop.vect(y_struct_one)

    # Ensure that each wedge of the modified wedge - original is
    # equal to 2d array of ones
    apply_along_wedges(
        Cop.struct(y_one - y),
        lambda c, w, s, na, ns: np.testing.assert_allclose(c, np.ones_like(c)),
    )


def test_energy():
    ndim = np.random.randint(1, 10)
    shape = [np.random.randint(1, 10) for _ in range(ndim)]
    ones = np.ones(shape)
    e = energy(ones)
    np.testing.assert_allclose(1.0, e)


def test_energy_split():
    shape = [np.random.randint(1, 100), np.random.randint(1, 100)]
    rows, cols = np.random.randint(1, shape[0]), np.random.randint(1, shape[1])
    ones = np.ones(shape)
    e = energy_split(ones, rows, cols)
    for row in range(rows):
        for col in range(cols):
            np.testing.assert_allclose(1.0, e[row][col])


def test_ndargmax():
    ndim = np.random.randint(1, 10)
    shape = [np.random.randint(1, 10) for _ in range(ndim)]
    ary = np.zeros(shape)
    index = tuple([np.random.randint(0, shape[i]) for i in range(ndim)])
    ary[index] = 1.0
    assert index == ndargmax(ary)
