import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from CaImaging.util import check_attrs


def overlay_footprints(ScrollObj):
    attrs = ["footprints"]
    check_attrs(ScrollObj, attrs)

    footprints = ScrollObj.footprints
    dims = footprints[0][0].shape
    overlay = np.zeros((dims[0], dims[1], 3))

    for channel, footprints_this_day in enumerate(footprints):
        footprint = footprints_this_day[ScrollObj.current_position]
        overlay[:, :, channel] = footprint / np.max(footprint)

    ax = ScrollObj.ax
    ax.imshow(overlay, origin="lower")

    ScrollObj.last_position = footprints[0].shape[0]


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    """
    Line show_plot with error bars except the error bars are filled in
    rather than the monstrosity from matplotlib.

    :parameters
    ---
    x: array-like
        x-axis values.

    y: array-like, same length as x
        y-axis values.

    yerr: array-like, same length as x and y
        Error around the y values.
    """
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)