"""Utility functions to help plotting."""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
import warnings


import astropy.units as u
from astropy.visualization import (
    PercentileInterval,
    ImageNormalize,
    SqrtStretch,
    LinearStretch,
)


def plot_image(
    image,
    ax=None,
    scale="linear",
    origin="lower",
    xlabel="Pixel Column Number",
    ylabel="Pixel Row Number",
    clabel="Flux ($e^{-}s^{-1}$)",
    title=None,
    show_colorbar=True,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Utility function to plot a 2D image

    Parameters
    ----------
    image : 2d array
        Image data.
    ax : `~matplotlib.axes.Axes`
        A matplotlib axes object to plot into. If no axes is provided,
        a new one will be generated.
    scale : str
        Scale used to stretch the colormap.
        Options: 'linear', 'sqrt', or 'log'.
    origin : str
        The origin of the coordinate system.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    clabel : str
        Label for the color bar.
    title : str or None
        Title for the plot.
    show_colorbar : bool
        Whether or not to show the colorbar
    vmin : float
        Minimum colorbar value. By default, the 2.5%-percentile is used.
    vmax : float
        Maximum colorbar value. By default, the 97.5%-percentile is used.
    kwargs : dict
        Keyword arguments to be passed to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The matplotlib axes object.
    """
    if isinstance(image, u.Quantity):
        image = image.value
    if ax is None:
        _, ax = plt.subplots()

    if vmin is None or vmax is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # ignore image NaN values
            mask = np.nan_to_num(image) > 0
            if mask.any() > 0:
                vmin_default, vmax_default = PercentileInterval(95.0).get_limits(
                    image[mask]
                )
            else:
                vmin_default, vmax_default = 0, 0
            if vmin is None:
                vmin = vmin_default
            if vmax is None:
                vmax = vmax_default

    norm = None
    if scale is not None:
        if scale == "linear":
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=False
            )
        elif scale == "sqrt":
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax, stretch=SqrtStretch(), clip=False
            )
        elif scale == "log":
            # To use log scale we need to guarantee that vmin > 0, so that
            # we avoid division by zero and/or negative values.
            norm = LogNorm(vmin=max(vmin, sys.float_info.epsilon), vmax=vmax, clip=True)
        else:
            raise ValueError("scale {} is not available.".format(scale))
    cax = ax.imshow(image, origin=origin, norm=norm, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_colorbar:
        cbar = plt.colorbar(cax, ax=ax, label=clabel)
        cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
        cbar.ax.minorticks_off()
    return ax
