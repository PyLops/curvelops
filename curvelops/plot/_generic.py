__all__ = [
    "create_colorbar",
    "create_axes_grid",
    "create_inset_axes_grid",
    "overlay_arrows",
]
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray


def _create_range(start, end, n):
    return start + (end - start) * (0.5 + np.arange(n)) / n


def create_colorbar(
    im: AxesImage,
    ax: Axes,
    size: float = 0.05,
    pad: float = 0.1,
    orientation: str = "vertical",
) -> Tuple[Axes, Colorbar]:
    r"""Create a colorbar.

    Divides  axis and attaches a colorbar to it.

    Parameters
    ----------
    im : :obj:`AxesImage <matplotlib.image.AxesImage>`
        Image from which the colorbar will be created.
        Commonly the output of :obj:`matplotlib.pyplot.imshow`.
    ax : :obj:`Axes <matplotlib.axes.Axes>`
        Axis which to split.
    size : :obj:`float`, optional
        Size of split, by default 0.05. Effectively sets the size of the colorbar.
    pad : :obj:`float`, optional`
        Padding between colorbar axis and input axis, by default 0.1.
    orientation : :obj:`str`, optional
        Orientation of the colorbar, by default "vertical".

    Returns
    -------
    Tuple[:obj:`Axes <matplotlib.axes.Axes>`, :obj:`Colorbar <matplotlib.colorbar.Colorbar>`]
        **cax** : Colorbar axis.

        **cb** : Colorbar.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.ticker import MultipleLocator
    >>> from curvelops.plot import create_colorbar
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow([[0]], vmin=-1, vmax=1, cmap="gray")
    >>> cax, cb = create_colorbar(im, ax)
    >>> cax.yaxis.set_major_locator(MultipleLocator(0.1))
    >>> print(cb.vmin)
    -1.0
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{size:%}", pad=pad)
    cb = ax.get_figure().colorbar(im, cax=cax, orientation=orientation)
    return cax, cb


def create_axes_grid(
    rows: int,
    cols: int,
    kwargs_figure: Optional[dict] = None,
    kwargs_gridspec: Optional[dict] = None,
    kwargs_subplots: Optional[dict] = None,
) -> Tuple[Figure, NDArray]:
    r"""Creates a grid of axes.

    Parameters
    ----------
    rows : :obj:`int`
        Number of rows.
    cols : :obj:`int`
        Number of columns.
    kwargs_figure : ``Optional[dict]``, optional
        Arguments to be passed to :obj:`matplotlib.pyplot.figure`.
    kwargs_gridspec : ``Optional[dict]``, optional
        Arguments to be passed to :obj:`matplotlib.gridspec.GridSpec`.
    kwargs_subplots : ``Optional[dict]``, optional
        Arguments to be passed to :obj:`matplotlib.figure.Figure.add_subplot`.


    Returns
    -------
    Tuple[:obj:`Figure <matplotlib.figure.Figure>`, :obj:`NDArray <numpy.typing.NDArray>`]
        **fig** : Figure.

        **axs** : Array of :obj:`Axes <matplotlib.axes.Axes>` shaped ``(rows, cols)``.

    Examples
    --------
    >>> from curvelops.plot import create_axes_grid
    >>> rows, cols = 2, 3
    >>> fig, axs = create_axes_grid(
    >>>     rows,
    >>>     cols,
    >>>     kwargs_figure=dict(figsize=(8, 8)),
    >>>     kwargs_gridspec=dict(wspace=0.3, hspace=0.3),
    >>> )
    >>> for irow in range(rows):
    >>>     for icol in range(cols):
    >>>         axs[irow][icol].plot(np.cos((2 + irow + icol**2) * np.linspace(0, 1)))
    >>>         axs[irow][icol].set(title=f"Row, Col: ({irow}, {icol})")
    """
    if kwargs_figure is None:
        kwargs_figure = {}
    if kwargs_gridspec is None:
        kwargs_gridspec = {}
    if kwargs_subplots is None:
        kwargs_subplots = {}
    fig = plt.figure(**kwargs_figure)
    grid = fig.add_gridspec(rows, cols, **kwargs_gridspec)
    axs = np.empty((rows, cols), dtype=Axes)
    for irow in range(rows):
        for icol in range(cols):
            axs[irow, icol] = fig.add_subplot(grid[irow, icol], **kwargs_subplots)
    return fig, axs


def create_inset_axes_grid(
    ax: Axes,
    rows: int,
    cols: int,
    height: float = 0.5,
    width: float = 0.5,
    kwargs_inset_axes: Optional[dict] = None,
) -> NDArray:
    r"""Create a grid of insets.

    The input axis will be overlaid with a grid of insets.
    Numbering of the axes is top to bottom (rows) and
    left to right (cols).

    Parameters
    ----------
    ax : :obj:`Axes <matplotlib.axes.Axes>`
        Input axis.
    rows : :obj:`int`
        Number of rows.
    cols : :obj:`int`
        Number of columns.
    width : :obj:`float`, optional
        Width of each axis, as a percentage of ``cols``, by default 0.5.
    height : :obj:`float`, optional
        Height of each axis, as a percentage of ``rows``, by default 0.5.
    kwargs_inset_axes : ``Optional[dict]``, optional
        Arguments to be passed to :obj:`matplotlib.axes.Axes.inset_axes`.

    Returns
    -------
    :obj:`NDArray <numpy.typing.NDArray>`
        Array of :obj:`Axes <matplotlib.axes.Axes>` shaped ``(rows, cols)``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from curvelops.plot import create_inset_axes_grid
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> ax.imshow([[0]], extent=[-2, 2, 2, -2], vmin=-1, vmax=1, cmap="gray")
    >>> rows, cols = 2, 3
    >>> inset_axes = create_inset_axes_grid(
    >>>     ax,
    >>>     rows,
    >>>     cols,
    >>>     width=0.5,
    >>>     height=0.5,
    >>>     kwargs_inset_axes=dict(projection="polar"),
    >>> )
    >>> nscales = 4
    >>> lw = 0.1
    >>> for irow in range(rows):
    >>>     for icol in range(cols):
    >>>         for iscale in range(1, nscales):
    >>>             inset_axes[irow][icol].bar(
    >>>                 x=0,
    >>>                 height=lw,
    >>>                 width=2 * np.pi,
    >>>                 bottom=((iscale + 1) - 0.5 * lw) / (nscales - 1),
    >>>                 color="r",
    >>>             )
    >>>             inset_axes[irow][icol].set(title=f"Row, Col: ({irow}, {icol})")
    >>>             inset_axes[irow][icol].axis("off")
    """
    if kwargs_inset_axes is None:
        kwargs_inset_axes = {}

    axes = np.empty((rows, cols), dtype=object)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    width *= (xmax - xmin) / cols
    height *= (ymax - ymin) / rows

    for irow, rowpos in enumerate(_create_range(ymin, ymax, rows)):
        for icol, colpos in enumerate(_create_range(xmin, xmax, cols)):
            axes[irow, icol] = ax.inset_axes(
                [colpos - 0.5 * width, rowpos - 0.5 * height, width, height],
                transform=ax.transData,
                **kwargs_inset_axes,
            )
    return axes


def overlay_arrows(
    vectors: NDArray, ax: Axes, arrowprops: Optional[dict] = None
) -> None:
    r"""Overlay arrows on an axis.

    Parameters
    ----------
    vectors : :obj:`NDArray <numpy.typing.NDArray>`
        Array shaped ``(rows, cols, 2)``, corresponding to a 2D vector field.
    ax : :obj:`Axes <matplotlib.axes.Axes>`
        Axis on which to overlay the arrows.
    arrowprops : ``Optional[dict]``, optional
        Arrow properties, to be passed to :obj:`matplotlib.pyplot.annotate`.
        By default will be set to ``dict(facecolor="black", shrink=0.05)``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from curvelops.plot import overlay_arrows
    >>> fig, ax = plt.subplots(figsize=(8, 10))
    >>> ax.imshow([[0]], vmin=-1, vmax=1, extent=[0, 1, 1, 0], cmap="gray")
    >>> rows, cols = 3, 4
    >>> kvecs = np.array(
    >>>     [
    >>>         [(1 + x, x * y) for x in (0.5 + np.arange(cols)) / cols]
    >>>         for y in (0.5 + np.arange(rows)) / rows
    >>>     ]
    >>> )
    >>> overlay_arrows(
    >>>     0.05 * kvecs,
    >>>     ax,
    >>>     arrowprops=dict(
    >>>         facecolor="r",
    >>>         shrink=0.05,
    >>>         width=10 / cols,
    >>>         headwidth=10,
    >>>         headlength=10,
    >>>     ),
    >>> )
    """
    rows, cols, _ = vectors.shape

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    if arrowprops is None:
        arrowprops = dict(facecolor="black", shrink=0.05)

    for irow, rowpos in enumerate(_create_range(ymin, ymax, rows)):
        for icol, colpos in enumerate(_create_range(xmin, xmax, cols)):
            ax.annotate(
                "",
                xy=(
                    colpos + vectors[irow, icol, 0],
                    rowpos + vectors[irow, icol, 1],
                ),
                xytext=(colpos, rowpos),
                xycoords="data",
                arrowprops=arrowprops,
                annotation_clip=False,
            )
