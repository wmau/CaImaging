from util import open_minian, read_eztrack
from util import synchronize_time_series as sync
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
from scipy.stats import binned_statistic_2d

class PlaceFields():
    def __init__(self, x, y, neural_data, bin_size_cm=20):
        """
        Place field object.

        :parameters
        ---
        x, y: (t,) arrays, positions per sample.
        neural_data: (n,t) array, neural activity.
        """
        self.x, self.y = x, y
        self.neural_data = neural_data

        self.make_occupancy_map(bin_size_cm=bin_size_cm, plot=False)


    def plot_dots(self, neuron, std_thresh=2, pos_color='k',
                  transient_color='r'):
        """
        Plots a dot plot. Position samples with suprathreshold activity
        dots overlaid.

        :parameters
        ---
        neuron: int, neuron index in neural_data.
        std_thresh: float, number of standard deviations above the mean
            to plot "spike" dot.
        pos_color: color-like, color to make position samples.
        transient_color: color-like, color to make calcium transient-associated
            position samples.

        """
        # Define threshold.
        thresh = np.mean(self.neural_data[neuron]) + \
                 std_thresh*np.std(self.neural_data[neuron])
        supra_thresh = self.neural_data[neuron] > thresh

        # Plot.
        plt.scatter(self.x, self.y, s=3, c=pos_color)
        plt.scatter(self.x[supra_thresh], self.y[supra_thresh],
                    s=3, c=transient_color)



    def bin(self, x, y, bin_size_cm=20, plot=True, weights=None):
        """
        Spatially bins the position data.

        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        plot: bool, flag for plotting.

        :returns
        ---
        H: (nx,ny) array, 2d histogram of position.
        xedges, yedges: (n+1,) array, bin edges along each dimension.
        """
        # Calculate the min and max of position.
        x_extrema = [min(self.x), max(self.x)]
        y_extrema = [min(self.y), max(self.y)]

        # Make bins.
        xbins = np.linspace(x_extrema[0], x_extrema[1],
                            np.round(np.diff(x_extrema)/bin_size_cm))
        ybins = np.linspace(y_extrema[0], y_extrema[1],
                            np.round(np.diff(y_extrema)/bin_size_cm))

        # Do the binning.
        H, xedges, yedges = np.histogram2d(y, x,
                                           [ybins, xbins],
                                           weights=weights)

        # Plot.
        if plot:
            plt.imshow(H)

        return H

    def make_occupancy_map(self, bin_size_cm=20, plot=True):
        """
        Makes the occupancy heat map of the animal.

        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        plot: bool, flag for plotting.

        """
        self.occupancy_map = self.bin(self.x, self.y,
                                      bin_size_cm=bin_size_cm, plot=plot)

        if plot:
            plt.imshow(self.occupancy_map, origin='lower')

    def make_place_field(self, neuron, bin_size_cm=20, plot=True):
        """
        Bins activity in space. Essentially a 2d histogram weighted by
        neural activity.

        :parameters
        ---
        neuron: int, neuron index in neural_data.
        bin_size_cm: float, bin size in centimeters.
        plot: bool, flag for plotting.

        :return
        ---
        pf: (x,y) array, 2d histogram of position weighted by activity.
        """
        pf, xedges, yedges = \
            self.bin(self.x, self.y, bin_size_cm=bin_size_cm, plot=False,
                     weights=self.neural_data[neuron])

        # Normalize by occupancy.
        pf = pf / self.occupancy_map
        if plot:
            plt.imshow(pf, origin='lower')

        return pf


if __name__ == '__main__':
    dpath = r'D:\Projects\GTime\Data\G123\2\H14_M46_S20'
    bpath = os.path.join(r'D:\Projects\GTime\Data\G123\2\H14_M46_S20\Behavior',
                         'Merged_tracked.csv')
    minian = open_minian(dpath)

    position = read_eztrack(bpath)
    position = sync(position, np.asarray(minian.S))

    P = PlaceFields(position['x'], position['y'], np.asarray(minian.S))
    P.plot_dots(1)