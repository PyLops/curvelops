/* fdct3d_wrapper (Pybind11 wrapper for Fast 3D Curvelet Wrapping Transform)
   Copyright (C) 2020-2023 Carlos Alberto da Costa Filho

    ${CXX} -O3 -Wall -shared -std=c++11 -fPIC \
        -I${FFTW}/fftw `python3 -m pybind11 --includes` \
        fdct2d_wrapper.cpp ${FDCT}/fdct3d/src/libfdct3d.a \
        -L${FFTW}/fftw/.libs -lfftw \
        -o fdct2d_wrapper`python3-config --extension-suffix`
*/

#include "fdct3d.hpp"
#include "fdct3dinline.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>
#include <sstream>
namespace py = pybind11;

py::tuple fdct3d_param_wrap(int m, int n, int p, int nbscales, int nbangles_coarse, int ac)
{
    // Almost sure this function creates a copy, but it's ok since the outputs are small
    vector<vector<double>> fxs, fys, fzs;
    vector<vector<int>> nxs, nys, nzs;
    fdct3d_param(m, n, p, nbscales, nbangles_coarse, ac, fxs, fys, fzs, nxs, nys, nzs);
    return py::make_tuple(fxs, fys, fzs, nxs, nys, nzs);
}

vector<vector<py::array_t<cpx>>> fdct3d_forward_wrap(int nbscales, int nbangles_coarse, int ac, py::array_t<cpx> x)
{
    // Our wrapper takes a NumPy array, but ``fdct3d_forward`` requires a CpxNumTns
    // input (which will be accessed read-only). So we must create CpxNumTns ``xtns``
    // which will "mirror" our input ``x`` in a no-copy fashion.
    // We also need to output ``c`` whose conversion to a Python list of lists of
    // CpxNumTns will be handled by pybind11. The vector -> list casting is automatic
    // in pybind11, whereas the CpxNumTns -> py::array_t<cpx> casting is inside our function.
    CpxNumTns xtns;
    vector<vector<CpxNumTns>> ctns;

    // Responsibly access py::array with possible casting to complex. See:
    // https://stackoverflow.com/questions/42645228/cast-numpy-array-to-from-custom-c-matrix-class-using-pybind11
    // Note: CurveLab uses Fortran-style indexing, so we must transpose the input array. We do this
    //       by simply reading it as a Fortran array
    auto buf = py::array_t<cpx, py::array::f_style | py::array::forcecast>::ensure(x);
    if (!buf)
        throw std::runtime_error("x array buffer is empty. If you're calling from Python this should not happen!");
    if (buf.ndim() != 3)
        throw std::runtime_error("x.ndims != 3");

    // We don't to initialize x(m, n, p) because this allocates an array on the heap!
    xtns._m = buf.shape()[0];
    xtns._n = buf.shape()[1];
    xtns._p = buf.shape()[2];
    xtns._data = (cpx *)buf.data(); // Put our Python array buffer pointer as the CpxNumTns data

    // Call our forward function with all the right types
    fdct3d_forward(xtns._m, xtns._n, xtns._p, nbscales, nbangles_coarse, ac, xtns, ctns);

    // Clear the structure as if it had never existed...
    // xtns didn't allocate any data, so we make sure it doesn't deallocate any on the way out
    xtns._m = xtns._n = xtns._p = 0;
    xtns._data = NULL;

    vector<vector<py::array_t<cpx>>> c;
    // Expand ``c`` to fit the scales
    c.resize(ctns.size());
    for (size_t i = 0; i < ctns.size(); i++)
    {
        // Now we expand each scale to fit the angles
        c[i].resize(ctns[i].size());
        for (size_t j = 0; j < ctns[i].size(); j++)
        {
            // Create capsule linked to `ctns[i][j]._data` to track its lifetime
            // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
            py::capsule free_when_done(
                ctns[i][j].data(),
                [](void *cpx_ptr)
                {
                    cpx *cpx_arr = reinterpret_cast<cpx *>(cpx_ptr);
                    delete[] cpx_arr;
                });

            py::array c_arr(
                {ctns[i][j]._p, ctns[i][j]._n, ctns[i][j]._m}, // Shape
                {sizeof(cpx) * ctns[i][j]._m * ctns[i][j]._n,  // Strides (in bytes) of the underlying data array
                 sizeof(cpx) * ctns[i][j]._m,
                 sizeof(cpx)},
                ctns[i][j].data(), // Data pointer
                free_when_done);

            c[i][j] = c_arr;
            ctns[i][j]._m = ctns[i][j]._n = ctns[i][j]._p = 0;
            ctns[i][j]._data = NULL;
        }
    }
    return c;
}

py::array_t<cpx> fdct3d_inverse_wrap(int m, int n, int p, int nbscales, int nbangles_coarse, int ac,
                                     vector<vector<py::array_t<cpx>>> c)
{
    // Similarly to the forward wrapper, we create ``ctns`` and ``xtns`` to use
    // as dummy input and output arrays.
    size_t i, j;
    CpxNumTns xtns;
    vector<vector<CpxNumTns>> ctns;

    if ((size_t)nbscales != c.size())
        throw std::runtime_error("nbscales != len(c)");

    // We copy the ``c`` "structure" onto a ``ctns`` "structure"
    // Start by expanding the first index of ``ctns`` to fit all scales
    ctns.resize(c.size());
    for (i = 0; i < c.size(); i++)
    {
        // Now we expand each scale to fit all angles for that scale
        ctns[i].resize(c[i].size());
        for (j = 0; j < c[i].size(); j++)
        {
            // Now we must copy the structure over to ``ctns``
            py::buffer_info buf = c[i][j].request();
            ctns[i][j]._m = buf.shape[2];
            ctns[i][j]._n = buf.shape[1];
            ctns[i][j]._p = buf.shape[0];
            ctns[i][j]._data = static_cast<cpx *>(buf.ptr);
        }
    }
    // No bounds checking is made inside this, so if ``c`` (or equivalently ``ctns``)
    // are not compatible with the other parameters, this function WILL segfault
    // TODO: Optionally sanitize this by calling ``fdct3d_param_wrap``
    fdct3d_inverse(m, n, p, nbscales, nbangles_coarse, ac, ctns, xtns);

    // Clear input structure without deallocating
    for (i = 0; i < c.size(); i++)
        for (j = 0; j < c[i].size(); j++)
        {
            ctns[i][j]._m = ctns[i][j]._n = ctns[i][j]._p = 0;
            ctns[i][j]._data = NULL;
        }

    py::capsule free_when_done(
        xtns.data(),
        [](void *cpx_ptr)
        {
            cpx *cpx_arr = reinterpret_cast<cpx *>(cpx_ptr);
            delete[] cpx_arr;
        });

    // Create output array
    py::array x({m, n, p},
                {sizeof(cpx),
                 sizeof(cpx) * m,
                 sizeof(cpx) * m * n},
                xtns.data(),
                free_when_done);

    // Clear output structure without deallocating
    xtns._m = xtns._n = xtns._p = 0;
    xtns._data = NULL;

    return x;
}

PYBIND11_MODULE(fdct3d_wrapper, m)
{
    m.doc() = "FDCT3D pybind11 wrapper";
    m.def("fdct3d_param_wrap", &fdct3d_param_wrap, "Parameters for 3D FDCT",
          py::return_value_policy::take_ownership);
    m.def("fdct3d_forward_wrap", &fdct3d_forward_wrap, "3D Forward FDCT",
          py::return_value_policy::take_ownership);
    m.def("fdct3d_inverse_wrap", &fdct3d_inverse_wrap, "3D Inverse FDCT",
          py::return_value_policy::take_ownership);
}
