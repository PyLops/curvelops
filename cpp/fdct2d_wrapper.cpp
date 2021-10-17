/* fdct2d_wrapper (Pybind11 wrapper for Fast 2D Curvelet Wrapping Transform)
   Copyright (C) 2020 Carlos Alberto da Costa Filho

    ${CXX} -O3 -Wall -shared -std=c++11 -fPIC \
        -I${FFTW}/fftw `python3 -m pybind11 --includes` \
        fdct2d_wrapper.cpp ${FDCT}/fdct_wrapping_cpp/src/libfdct_wrapping.a \
        -L${FFTW}/fftw/.libs -lfftw \
        -o fdct2d_wrapper`python3-config --extension-suffix`
*/

#include "fdct_wrapping.hpp"
#include "fdct_wrapping_inline.hpp"
#include "fdct_wrapping_inc.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>
#include <sstream>
namespace py = pybind11;
using namespace std;
using namespace fdct_wrapping_ns;

py::tuple fdct2d_param_wrap(int m, int n, int nbscales, int nbangles_coarse, int ac)
{
    // Almost sure this function creates a copy, but it's ok since the outputs are small
    vector<vector<double>> sx, sy;
    vector<vector<double>> fx, fy;
    vector<vector<int>> nx, ny;
    fdct_wrapping_param(m, n, nbscales, nbangles_coarse, ac, sx, sy, fx, fy, nx, ny);
    return py::make_tuple(sx, sy, fx, fy, nx, ny);
}

vector<vector<py::array_t<cpx>>> fdct2d_forward_wrap(int nbscales, int nbangles_coarse, int ac, py::array_t<cpx> x)
{
    // Our wrapper takes a NumPy array, but ``fdct_wrapping`` requires a CpxNumMat
    // input (which will be accessed read-only). So we must create CpxNumMat ``xmat``
    // which will "mirror" our input ``x`` in a no-copy fashion.
    // We also need to output ``c`` whose conversion to a Python list of lists of
    // CpxNumMat will be handled by pybind11. The vector -> list casting is automatic
    // in pybind11, whereas the CpxNumMat -> py::array_t<cpx> casting is inside our function.
    CpxNumMat xmat;
    vector<vector<CpxNumMat>> cmat;

    // Responsibly access py::array with possible casting to complex. See:
    // https://stackoverflow.com/questions/42645228/cast-numpy-array-to-from-custom-c-matrix-class-using-pybind11
    // Note: CurveLab uses Fortran-style indexing, so we must transpose the input array. We do this
    //       by simply reading it as a Fortran array
    auto buf = py::array_t<cpx, py::array::f_style | py::array::forcecast>::ensure(x);
    if (!buf)
        throw std::runtime_error("x array buffer is empty. If you're calling from Python this should not happen!");
    if (buf.ndim() != 2)
        throw std::runtime_error("x.ndims != 2");

    // We don't to initialize ``x(m, n)`` because this allocates an array on the heap!
    xmat._m = buf.shape()[0];
    xmat._n = buf.shape()[1];
    xmat._data = (cpx *)buf.data(); // Put our Python array buffer pointer as the CpxNumMat data

    // Call our forward function with all the right types
    fdct_wrapping(xmat._m, xmat._n, nbscales, nbangles_coarse, ac, xmat, cmat);

    // Clear the structure as if it had never existed...
    // xmat didn't allocate any data, so we make sure it doesn't deallocate any on the way out
    xmat._m = xmat._n = 0;
    xmat._data = NULL;

    vector<vector<py::array_t<cpx>>> c;
    // Expand ``c`` to fit the scales
    c.resize(cmat.size());
    for (size_t i = 0; i < cmat.size(); i++)
    {
        // Now we expand each scale to fit the angles
        c[i].resize(cmat[i].size());
        for (size_t j = 0; j < cmat[i].size(); j++)
        {
            // Create capsule linked to `cmat[i][j]._data` to track its lifetime
            // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
            py::capsule free_when_done(
                cmat[i][j].data(),
                [](void *cpx_ptr)
                {
                    cpx *cpx_arr = reinterpret_cast<cpx *>(cpx_ptr);
                    delete[] cpx_arr;
                });

            py::array c_arr(
                {cmat[i][j]._n, cmat[i][j]._m}, // Shape
                {sizeof(cpx) * cmat[i][j]._m,   // Strides (in bytes) of the underlying data array
                 sizeof(cpx)},
                cmat[i][j].data(), // Data pointer
                free_when_done);

            c[i][j] = c_arr;
            cmat[i][j]._m = cmat[i][j]._n = 0;
            cmat[i][j]._data = NULL;
        }
    }
    return c;
}

py::array_t<cpx> fdct2d_inverse_wrap(int m, int n, int nbscales, int nbangles_coarse, int ac,
                                     vector<vector<py::array_t<cpx>>> c)
{
    // Similarly to the forward wrapper, we create ``cmat`` and ``xmat`` to use
    // as dummy input and output arrays.
    size_t i, j;
    CpxNumMat xmat;
    vector<vector<CpxNumMat>> cmat;

    if ((size_t)nbscales != c.size())
        throw std::runtime_error("nbscales != len(c)");

    // We copy the ``c`` "structure" onto a ``cmat`` "structure"
    // Start by expanding the first index of ``cmat`` to fit all scales
    cmat.resize(c.size());
    for (i = 0; i < c.size(); i++)
    {
        // Now we expand each scale to fit all angles for that scale
        cmat[i].resize(c[i].size());
        for (j = 0; j < c[i].size(); j++)
        {
            // Now we must copy the structure over to ``cmat``
            py::buffer_info buf = c[i][j].request();
            cmat[i][j]._m = buf.shape[1];
            cmat[i][j]._n = buf.shape[0];
            cmat[i][j]._data = static_cast<cpx *>(buf.ptr);
        }
    }
    // No bounds checking is made inside this, so if ``c`` (or equivalently ``cmat``)
    // are not compatible with the other parameters, this function WILL segfault
    // TODO: Optionally sanitize this by calling ``fdct2d_param_wrap``
    ifdct_wrapping(m, n, nbscales, nbangles_coarse, ac, cmat, xmat);

    // Clear input structure without deallocating
    for (i = 0; i < c.size(); i++)
        for (j = 0; j < c[i].size(); j++)
        {
            cmat[i][j]._m = cmat[i][j]._n = 0;
            cmat[i][j]._data = NULL;
        }

    py::capsule free_when_done(
        xmat.data(),
        [](void *cpx_ptr)
        {
            cpx *cpx_arr = reinterpret_cast<cpx *>(cpx_ptr);
            delete[] cpx_arr;
        });

    // Create output array
    py::array x({m, n},
                {sizeof(cpx),
                 sizeof(cpx) * m},
                xmat.data(),
                free_when_done);

    // Clear output structure without deallocating
    xmat._m = xmat._n = 0;
    xmat._data = NULL;

    return x;
}

PYBIND11_MODULE(fdct2d_wrapper, m)
{
    m.doc() = "FDCT2D pybind11 wrapper";
    m.def("fdct2d_param_wrap", &fdct2d_param_wrap, "Parameters for 2D FDCT",
          py::return_value_policy::take_ownership);
    m.def("fdct2d_forward_wrap", &fdct2d_forward_wrap, "2D Forward FDCT",
          py::return_value_policy::take_ownership);
    m.def("fdct2d_inverse_wrap", &fdct2d_inverse_wrap, "2D Inverse FDCT",
          py::return_value_policy::take_ownership);
}
