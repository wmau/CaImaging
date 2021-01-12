import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import check_attrs

def overlay_footprints(ScrollObj):
    attrs = ['footprints']
    check_attrs(ScrollObj, attrs)

    footprints = ScrollObj.footprints
    dims = footprints[0][0].shape
    overlay = np.zeros((dims[0], dims[1], 3))

    for channel, footprints_this_day in enumerate(footprints):
        footprint = footprints_this_day[ScrollObj.current_position]
        overlay[:,:,channel] = footprint / np.max(footprint)

    ax = ScrollObj.ax
    ax.imshow(overlay, origin='lower')

    ScrollObj.last_position = footprints[0].shape[0]
