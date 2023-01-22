__all__ = [
    "curveshow",
    "overlay_disks",
]
import itertools
from math import ceil, floor
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from curvelops import apply_along_wedges, energy_split

from ..typing import FDCTStructLike


def curveshow(
    c_struct: FDCTStructLike,
    k_space: bool = False,
    basesize: int = 5,
    showaxis: bool = False,
    kwargs_imshow: Optional[dict] = None,
) -> List[Figure]:
    """Display curvelet coefficients in each wedge as images.

    For each curvelet scale, display a figure with each wedge
    plotted as an image in its own axis.

    Parameters
    ----------
    c_struct : :obj:`FDCTStructLike <curvelops.typing.FDCTStructLike>`
        Curvelet structure.
    k_space :  :obj:`bool`, optional
        Show curvelet coefficient (False) or its 2D FFT transform (True),
        by default False.
    basesize : :obj:`int`, optional
        Base fize of figure, by default 5. Each figure will be sized
        ``(basesize * cols, basesize * rows)``, where
        ``rows = floor(sqrt(nangles))`` and ``cols = ceil(nangles / rows)``
    showaxis : :obj:`bool`, optional
        Turn on axis lines and labels, by default False.
    kwargs_imshow : ``Optional[dict]``, optional
        Arguments to be passed to :obj:`matplotlib.pyplot.imshow`.

    Examples
    --------
    >>> import numpy as np
    >>> from curvelops import FDCT2D
    >>> from curvelops.utils import apply_along_wedges, energy
    >>> from curvelops.plot import curveshow
    >>> d = np.random.randn(101, 101)
    >>> C = FDCT2D(d.shape, nbscales=2, nbangles_coarse=8)
    >>> y = C.struct(C @ d)
    >>> y_norm = apply_along_wedges(y, lambda w, *_: w / energy(w))
    >>> curveshow(
    >>>     y_norm,
    >>>     basesize=2,
    >>>     kwargs_imshow=dict(aspect="auto", vmin=-1, vmax=1, cmap="RdBu")
    >>> )

    Returns
    -------
    List[:obj:`Figure <matplotlib.figure.Figure>`]
        One figure per scale.
    """

    def fft(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

    _kwargs_imshow_default = {}
    if k_space:
        _kwargs_imshow_default["vmax"] = np.abs(fft(c_struct[0][0])).max()
        _kwargs_imshow_default["vmin"] = 0.0
        _kwargs_imshow_default["cmap"] = "turbo"
    else:
        _kwargs_imshow_default["vmax"] = np.abs(c_struct[0][0]).max()
        _kwargs_imshow_default["vmin"] = -_kwargs_imshow_default["vmax"]
        _kwargs_imshow_default["cmap"] = "gray"
    if kwargs_imshow is None:
        kwargs_imshow = _kwargs_imshow_default
    else:
        kwargs_imshow = {**_kwargs_imshow_default, **kwargs_imshow}

    figs_axes = []
    for iscale, c_scale in enumerate(c_struct):
        nangles = len(c_scale)
        rows = floor(np.sqrt(nangles))
        cols = ceil(nangles / rows)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(basesize * cols, basesize * rows),
        )
        fig.suptitle(
            f"Scale {iscale} ({nangles} wedge{'s' if nangles > 1 else ''})"
        )
        figs_axes.append((fig, axes))
        axes = np.atleast_1d(axes).ravel()

        for iwedge, (c_wedge, ax) in enumerate(zip(c_scale, axes)):
            if k_space:
                ax.imshow(np.abs(fft(c_wedge)), **kwargs_imshow)
            else:
                ax.imshow(c_wedge.real, **kwargs_imshow)
            if nangles > 1:
                ax.set(title=f"Wedge {iwedge}")
            if not showaxis:
                ax.axis("off")
            fig.tight_layout()
    return figs_axes


def overlay_disks(
    c_struct: FDCTStructLike,
    axes: NDArray,
    linewidth: float = 0.5,
    linecolor: str = "r",
    map_cmap: bool = True,
    cmap: Union[str, Colormap] = "gray_r",
    alpha: float = 1.0,
    pclip: float = 1.0,
    map_alpha: bool = False,
    min_alpha: float = 0.05,
    normalize: str = "all",
    annotate: bool = False,
):
    """Overlay curvelet disks over a 2D grid of axes.

    Its intended usage is to display the strength of curvelet coefficients
    of a certain image with a disk display. Given an ``axes`` 2D array,
    each curvelet wedge will be split into ``rows, cols = axes.shape``
    sub-wedges. The energy of each of these sub-wedges will be mapped
    to a colormap color and/or transparency.

    See Also
    --------
    :obj:`energy_split <curvelops.utils.energy_split>`: Splits a wedge into ``(rows, cols)`` wedges and computes the energy of each of these subdivisions.

    :obj:`create_inset_axes_grid`: Create a grid of insets.

    :obj:`create_axes_grid`: Creates a grid of axes.

    :obj:`curveshow`: Display curvelet coefficients in each wedge as images.

    Parameters
    ----------
    c_struct : :obj:`FDCTStructLike <curvelops.typing.FDCTStructLike>`:
        Curvelet coefficients of underlying image.
    axes : :obj:`NDArray <numpy.typing.NDArray>`
        2D grid of axes for which disks will be computed.
    linewidth : :obj:`float`, optional
        Width of line separating scales, by default 0.5.
        Will be scaled by ``0.1 / nscales`` internally.
        Set to zero to disable.
    linecolor : :obj:`str`, optional
        Color of line separating scales, by default "r".
    map_cmap : :obj:`bool`, optional
        When enabled, energy will be mapped to the colormap, by default True.
    cmap : Union[:obj:`str`, :obj:`Colormap <matplotlib.colors.Colormap>`], optional
        Colormap, by default ``"gray_r"``.
    alpha : :obj:`float`, optional
        When using ``map_cmap``, sets a transparecy for all wedges.
        Has no effect when ``map_alpha`` is enabled. By default 1.0.
    pclip : :obj:`float`, optional
        Clips the maximum amplitude by this percentage. By default 1.0.
        Should be between 0.0 and 1.0.
    map_alpha : :obj:`bool`, optional
        When enabled, energy will be mapped to the transparency, by default False.
    min_alpha : :obj:`float`, optional
        When using ``map_alpha``, sets a minimum transparency value.
        Has no effect when ``map_alpha`` is disabled. By default 0.05.
    normalize : :obj:`str`, optional
        Normalize wedges by:

        * ``"all"`` (default)
            Colormap/alpha value of 1.0 will correspond to the maximum
            energy found across all wedges

        * ``"scale"``
            Colormap/alpha value of 1.0 will correspond to the maximum
            energy found across all wedges in the same scale.
    annotate : :obj:`bool`, optional
        When true, will display in the middle of the wedge a
        pair of numbers ``iscale, iwedge``, the index of that scale
        and that wedge, both starting from zero. This option is useful to
        understand which directions each wedge corresponds to.
        By default False.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from curvelops import FDCT2D
    >>> from curvelops.utils import apply_along_wedges
    >>> from curvelops.plot import create_axes_grid, overlay_disks
    >>> x = np.random.randn(50, 100)
    >>> C = FDCT2D(x.shape, nbscales=4, nbangles_coarse=8)
    >>> y = C.struct(C @ x)
    >>> y_ones = apply_along_wedges(y, lambda w, *_: np.ones_like(w))
    >>> fig, axes = create_axes_grid(
    >>>     1,
    >>>     1,
    >>>     kwargs_subplots=dict(projection="polar"),
    >>>     kwargs_figure=dict(figsize=(8, 8)),
    >>> )
    >>> overlay_disks(y_ones, axes, annotate=True, cmap="gray")

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
    >>> from curvelops import FDCT2D
    >>> from curvelops.plot import create_inset_axes_grid, overlay_disks
    >>> from curvelops.utils import apply_along_wedges
    >>> plt.rcParams.update({"image.interpolation": "blackman"})
    >>> # Construct signal
    >>> xlim = [-1.0, 1.0]
    >>> ylim = [-0.5, 0.5]
    >>> x = np.linspace(*xlim, 201)
    >>> z = np.linspace(*ylim, 101)
    >>> xm, zm = np.meshgrid(x, z, indexing="ij")
    >>> freq = 5
    >>> d = np.cos(2 * np.pi * freq * (xm + np.cos(xm) * zm) ** 3)
    >>> # Compute curvelet coefficients
    >>> C = FDCT2D(d.shape, nbangles_coarse=8, allcurvelets=False)
    >>> d_c = C.struct(C @ d)
    >>> # Plot original signal
    >>> fig, ax = plt.subplots(figsize=(8, 4    ))
    >>> ax.imshow(d.T, extent=[*xlim, *(ylim[::-1])], cmap="RdYlBu", vmin=-1, vmax=1)
    >>> ax.axis("off")
    >>> # Overlay disks
    >>> rows, cols = 3, 6
    >>> axesin = create_inset_axes_grid(
    >>>     ax, rows, cols, width=0.75, kwargs_inset_axes=dict(projection="polar")
    >>> )
    >>> pclip = 0.2
    >>> cmap = "gray_r"
    >>> overlay_disks(d_c, axesin, linewidth=0.0, pclip=pclip, cmap=cmap)
    >>> # Display disk colorbar
    >>> divider = make_axes_locatable(ax)
    >>> cax = divider.append_axes("right", size="5%", pad=0.1)
    >>> mpl.colorbar.ColorbarBase(
    >>>     cax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=pclip)
    >>> )
    """
    rows, cols = axes.shape
    e_split = apply_along_wedges(
        c_struct, lambda w, *_: energy_split(w, rows, cols)
    )
    max_e = max(a.max() for a in itertools.chain.from_iterable(e_split))

    cmapper = cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, pclip, clip=True), cmap=cmap
    )

    nscales = len(c_struct)
    linewidth *= 0.1 / nscales

    for iscale in range(nscales):
        nangles = len(c_struct[iscale])
        angles_per_wedge = 2 * np.pi / nangles

        if normalize == "scale":
            max_e = max(
                a.max() for a in itertools.chain.from_iterable(e_split[iscale])
            )

        # To start starting counterclockwise from the top middle,
        # we need to offset the wedge index by the following amount
        iwedge_offset = nangles - nangles // 8
        for iwedge in range(nangles):
            for irow in range(rows):
                for icol in range(cols):
                    e = e_split[iscale][iwedge][irow, icol]
                    if map_alpha:
                        alpha = np.clip(
                            min_alpha + (1 - min_alpha) * e / max_e,
                            min_alpha,
                            1,
                        )
                    if map_cmap:
                        color = cmapper.to_rgba(np.clip(e / max_e, 0, 1))
                    else:
                        color = cmapper.to_rgba(1)

                    # Place the starting wedges in the correct place
                    iwedge_shift = (
                        nangles // 2 + iwedge + iwedge_offset
                    ) % nangles

                    # Wedge coordinates in polar plot
                    wedge_x = (iwedge_shift + 0.5) * angles_per_wedge
                    wedge_width = angles_per_wedge
                    wedge_height = 1 / (nscales - 1)
                    wedge_bottom = iscale * wedge_height
                    axes[irow][icol].bar(
                        x=wedge_x,
                        height=wedge_height,
                        width=wedge_width,
                        bottom=wedge_bottom,
                        color=color,
                        alpha=alpha,
                    )
                    if nangles > 1:
                        axes[irow][icol].bar(
                            x=wedge_x - wedge_width / 2,
                            height=wedge_height,
                            width=linewidth,
                            bottom=wedge_bottom,
                            color=linecolor,
                        )
                    if annotate:
                        axes[irow][icol].text(
                            wedge_x,
                            wedge_bottom
                            + (0 if wedge_bottom == 0 else wedge_height / 2),
                            f"{iscale}, {iwedge}",
                            backgroundcolor="w",
                            color="k",
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=6,
                        )

    # Plot line separating scales
    for irow in range(rows):
        for icol in range(cols):
            axes[irow][icol].axis("off")
            for iscale in range(nscales):
                if linewidth > 0.0:
                    axes[irow][icol].bar(
                        x=0,
                        height=linewidth,
                        width=2 * np.pi,
                        bottom=(iscale + 1 - linewidth / 2) / (nscales - 1),
                        color=linecolor,
                    )
