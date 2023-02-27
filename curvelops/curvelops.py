"""
``curvelops.curvelops``
=======================

Provides a LinearOperator for the 2D and 3D curvelet transforms.
"""

from itertools import product
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.core.multiarray import normalize_axis_index  # type: ignore
from numpy.typing import DTypeLike, NDArray
from pylops import LinearOperator
from pylops.utils.typing import InputDimsLike

from .fdct2d_wrapper import *  # noqa: F403
from .fdct3d_wrapper import *  # noqa: F403
from .typing import FDCTStructLike


def _fdct_docs(dimension: int) -> str:
    if dimension == 2:
        doc = "2D"
    elif dimension == 3:
        doc = "3D"
    else:
        doc = "2D/3D"
    return f"""{doc} dimensional Curvelet operator.
        Apply {doc} Curvelet Transform along two ``axes`` of a
        multi-dimensional array of size ``dims``.

        Parameters
        ----------
        dims : :obj:`tuple`
            Number of samples for each dimension.
        axes : :obj:`tuple`, optional
            Axes along which FDCT is applied.
        nbscales : :obj:`int`, optional
            Number of scales (including the coarsest level);
            Defaults to ceil(log2(min(input_dims)) - 3).
        nbangles_coarse : :obj:`int`, optional
            Number of angles at 2nd coarsest scale.
        allcurvelets : :obj:`bool`, optional
            Use curvelets at the finest (last) scale.
            If ``False``, a wavelet transform will be used for the
            finest scale. The coarsest scale is always a wavelet transform;
            the ones between the coarsest and the finest are all curvelet
            transforms. This option only affects the finest scale.
        dtype : :obj:`DTypeLike <numpy.typing.DTypeLike>`, optional
            ``dtype`` of the transform.
        """


class FDCT(LinearOperator):
    __doc__ = _fdct_docs(0)

    def __init__(
        self,
        dims: InputDimsLike,
        axes: Tuple[int, ...],
        nbscales: Optional[int] = None,
        nbangles_coarse: int = 16,
        allcurvelets: bool = True,
        dtype: DTypeLike = "complex128",
    ) -> None:
        ndim = len(dims)

        # Ensure axes are between 0, ndim-1
        axes = tuple(normalize_axis_index(d, ndim) for d in axes)

        # If input is shaped (100, 200, 300) and axes = (0, 2)
        # then input_shape will be (100, 300)
        self._input_shape = list(int(dims[d]) for d in axes)
        if nbscales is None:
            nbscales = int(np.ceil(np.log2(min(self._input_shape)) - 3))

        # Check dimension
        sizes: Union[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray, NDArray]]
        if len(axes) == 2:
            self.fdct: Callable = fdct2d_forward_wrap  # type: ignore  # noqa: F405
            self.ifdct: Callable = fdct2d_inverse_wrap  # type: ignore  # noqa: F405
            _, _, _, _, nxs, nys = fdct2d_param_wrap(  # type: ignore  # noqa: F405
                *self._input_shape, nbscales, nbangles_coarse, allcurvelets
            )
            sizes = (nys, nxs)
        elif len(axes) == 3:
            self.fdct: Callable = fdct3d_forward_wrap  # type: ignore  # noqa: F405
            self.ifdct: Callable = fdct3d_inverse_wrap  # type: ignore  # noqa: F405
            _, _, _, nxs, nys, nzs = fdct3d_param_wrap(  # type: ignore  # noqa: F405
                *self._input_shape, nbscales, nbangles_coarse, allcurvelets
            )
            sizes = (nzs, nys, nxs)
        else:
            raise NotImplementedError("FDCT is only implemented in 2D or 3D")

        # Complex operator is required to handle complex input
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            cpx = True
        else:
            cpx = False
            raise NotImplementedError("Only complex types supported")

        # Now we need to build the iterator which will only iterate along
        # the required axes. Following the example above,
        # iterable_axes = [ False, True, False ]
        iterable_axes = [i not in axes for i in range(ndim)]
        iterable_dims = np.array(dims)[iterable_axes]
        self._ndim_iterable = np.prod(iterable_dims)

        # Build the iterator itself. In our example, the slices
        # would be [:, i, :] for i in range(200)
        # We use slice(None) is the colon operator
        self._iterator = list(
            product(
                *(
                    range(dims[ax]) if doiter else [slice(None)]
                    for ax, doiter in enumerate(iterable_axes)
                )
            )
        )

        # For a single 2d/3d input, the length of the vector will be given by
        # the shapes in FDCT.sizes
        self.shapes = []
        for i in range(len(nxs)):
            shape = []
            for j in range(len(nxs[i])):
                shape.append(tuple(s[i][j] for s in sizes))
            self.shapes.append(shape)

        self._output_len = sum(np.prod(j) for i in self.shapes for j in i)

        # Save some useful properties
        self.inpdims = dims
        self.axes = axes
        self.nbscales = nbscales
        self.nbangles_coarse = nbangles_coarse
        self.allcurvelets = allcurvelets
        self.cpx = cpx

        # Required by PyLops
        super().__init__(
            dtype=dtype,
            dims=self.inpdims,
            dimsd=(*iterable_dims, self._output_len),
        )

    def _matvec(self, x: NDArray) -> NDArray:
        fwd_out = np.zeros((self._output_len, self._ndim_iterable), dtype=self.dtype)
        for i, index in enumerate(self._iterator):
            x_shaped = np.array(x.reshape(self.inpdims)[index])
            c_struct: FDCTStructLike = self.fdct(
                self.nbscales,
                self.nbangles_coarse,
                self.allcurvelets,
                x_shaped,
            )
            fwd_out[:, i] = self.vect(c_struct)
        return fwd_out.ravel()

    def _rmatvec(self, y: NDArray) -> NDArray:
        y_shaped = y.reshape(self._output_len, self._ndim_iterable)
        inv_out = np.zeros(self.inpdims, dtype=self.dtype)
        for i, index in enumerate(self._iterator):
            y_struct = self.struct(np.array(y_shaped[:, i]))
            xinv: NDArray = self.ifdct(
                *self._input_shape,
                self.nbscales,
                self.nbangles_coarse,
                self.allcurvelets,
                y_struct,
            )
            inv_out[index] = xinv

        return inv_out.ravel()

    def inverse(self, x: NDArray) -> NDArray:
        return self._rmatvec(x)

    def struct(self, x: NDArray) -> FDCTStructLike:
        """Convert curvelet flattened vector to curvelet structure.

        The FDCT always returns a 1D vector that has all curvelet
        coefficients. These coefficients can be organized into
        scales, wedges and spatial positions. Applying this
        function to a 1D vector generates this structure.

        Parameters
        ----------
        x : :obj:`NDArray <numpy.typing.NDArray>`
            Input flattened vector.

        Returns
        -------
        :obj:`FDCTStructLike <curvelops.typing.FDCTStructLike>`
            Curvelet structure, a list of lists of multidimensional arrays.
            The first index corresponds to scale, the second corresponds to
            angular wedge.
        """
        c_struct: FDCTStructLike = []
        k = 0
        for i in range(len(self.shapes)):
            angles = []
            for j in range(len(self.shapes[i])):
                size = np.prod(self.shapes[i][j])
                angles.append(x[k : k + size].reshape(self.shapes[i][j]))
                k += size
            c_struct.append(angles)
        return c_struct

    def vect(self, x: FDCTStructLike) -> NDArray:
        """Convert curvelet structure to curvelet flattened vector.

        The FDCT always returns a 1D vector that has all curvelet
        coefficients. These coefficients can be organized into
        scales, wedges and spatial positions. Applying this
        function to a curvelet structure returns the flattened
        vector.

        Parameters
        ----------
        x : :obj:`FDCTStructLike <curvelops.typing.FDCTStructLike>`
            Input curvelet structure.

        Returns
        -------
        :obj:`NDArray <numpy.typing.NDArray>`
            Flattened vector.
        """
        return np.concatenate([coef.ravel() for angle in x for coef in angle])


class FDCT2D(FDCT):
    __doc__ = _fdct_docs(2)

    def __init__(
        self,
        dims: InputDimsLike,
        axes: Tuple[int, ...] = (-2, -1),
        nbscales: Optional[int] = None,
        nbangles_coarse: int = 16,
        allcurvelets: bool = True,
        dtype: DTypeLike = "complex128",
    ) -> None:
        if len(axes) != 2:
            raise ValueError("FDCT2D must be called with exactly two axes")
        super().__init__(dims, axes, nbscales, nbangles_coarse, allcurvelets, dtype)


class FDCT3D(FDCT):
    __doc__ = _fdct_docs(3)

    def __init__(
        self,
        dims: InputDimsLike,
        axes: Tuple[int, ...] = (-3, -2, -1),
        nbscales: Optional[int] = None,
        nbangles_coarse: int = 16,
        allcurvelets: bool = True,
        dtype: DTypeLike = "complex128",
    ) -> None:
        if len(axes) != 3:
            raise ValueError("FDCT3D must be called with exactly three axes")
        super().__init__(dims, axes, nbscales, nbangles_coarse, allcurvelets, dtype)
