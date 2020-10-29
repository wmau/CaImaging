import numpy as np
from CaImaging.Behavior import spatial_bin
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 12})


class PlaceFields:
    def __init__(self, x, y, neural_data, bin_size=20, one_dim=False):
        """
        Place field object.

        :parameters
        ---
        x, y: (t,) arrays, positions per sample.
        neural_data: (n,t) array, neural activity.
        """
        self.x, self.y = x, y
        self.neural_data = neural_data
        self.one_dim = one_dim
        self.bin_size = bin_size

        self.make_occupancy_map(plot=False)

    def plot_dots(
        self, neuron, std_thresh=2, pos_color="k", transient_color="r", ax=None
    ):
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
        thresh = np.mean(self.neural_data[neuron]) + std_thresh * np.std(
            self.neural_data[neuron]
        )
        supra_thresh = self.neural_data[neuron] > thresh

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.x, self.y, s=3, c=pos_color)
        ax.scatter(self.x[supra_thresh], self.y[supra_thresh], s=3, c=transient_color)

    def make_occupancy_map(self, plot=True, ax=None):
        """
        Makes the occupancy heat cell_map of the animal.

        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        plot: bool, flag for plotting.

        """
        self.occupancy_map = spatial_bin(
            self.x, self.y, bin_size_cm=self.bin_size, plot=plot, one_dim=self.one_dim
        )[0]

        if plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(self.occupancy_map, origin="lower")

    def make_place_field(self, neuron, plot=True, ax=None):
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
        pf = spatial_bin(
            self.x,
            self.y,
            bin_size_cm=self.bin_size,
            plot=False,
            weights=self.neural_data[neuron],
            one_dim=self.one_dim,
        )[0]

        # Normalize by occupancy.
        pf = pf / self.occupancy_map
        if plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(pf, origin="lower")

        return pf


if __name__ == "__main__":
    mouse = "G132"
