import numpy as np
from CaImaging.Behavior import spatial_bin
import matplotlib.pyplot as plt
from random import randint
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent import futures

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 12})


class PlaceFields:
    def __init__(
        self, x, y, neural_data, bin_size=20, one_dim=False, shuffle_test=False
    ):
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

        self.occupancy_map = self.make_occupancy_map(show_plot=False)
        self.pfs = self.make_all_place_fields()
        self.spatial_information = [
            spatial_information(pf, self.occupancy_map) for pf in self.pfs
        ]
        self.pf_centers = self.find_pf_centers()
        if shuffle_test:
            self.pvals = self.assess_spatial_sig_parallel()

    def make_all_place_fields(self):
        """
        Compute the spatial rate maps of all neurons.

        :return:
        """
        pfs = []
        for neuron in range(self.neural_data.shape[0]):
            pfs.append(self.make_place_field(neuron, show_plot=False))

        return pfs

    def make_snake_plot(self, order='sorted', neurons='all', normalize=True):
        if neurons == 'all':
            neurons = [n for n in range(self.neural_data.shape[0])]
        pfs = self.pfs[neurons]

        if order == 'sorted':
            order = np.argsort(self.pf_centers[neurons])

        if normalize:
            pfs = pfs / pfs.max(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        ax.imshow(pfs[order])


    def find_pf_centers(self):
        centers = [np.argmax(pf) for pf in self.pfs]

        return centers

    def assess_spatial_sig(self, neuron, n_shuffles=500):
        shuffled_SIs = []
        for i in range(n_shuffles):
            shuffled_pf = self.make_place_field(neuron, show_plot=False, shuffle=True)
            shuffled_SIs.append(spatial_information(shuffled_pf, self.occupancy_map))

        p_value = np.sum(self.spatial_information[neuron] < shuffled_SIs) / n_shuffles

        return p_value

    def assess_spatial_sig_parallel(self):
        neurons = tqdm([n for n in range(self.neural_data.shape[0])])
        n_cores = mp.cpu_count()
        # with futures.ProcessPoolExecutor() as pool:
        #     results = pool.map(self.assess_spatial_sig, neurons)
        results = Parallel(n_jobs=n_cores)(
            delayed(self.assess_spatial_sig)(i) for i in neurons
        )

        return results

    def plot_dots(
        self, neuron, std_thresh=2, pos_color="k", transient_color="r", ax=None
    ):
        """
        Plots a dot show_plot. Position samples with suprathreshold activity
        dots overlaid.

        :parameters
        ---
        neuron: int, neuron index in neural_data.
        std_thresh: float, number of standard deviations above the mean
            to show_plot "spike" dot.
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

    def make_occupancy_map(self, show_plot=True, ax=None):
        """
        Makes the occupancy heat cell_map of the animal.

        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        show_plot: bool, flag for plotting.

        """
        occupancy_map = spatial_bin(
            self.x,
            self.y,
            bin_size_cm=self.bin_size,
            show_plot=show_plot,
            one_dim=self.one_dim,
        )[0]

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(occupancy_map, origin="lower")

        return occupancy_map

    def make_place_field(
        self, neuron, show_plot=True, normalize_by_occ=False, ax=None, shuffle=False
    ):
        """
        Bins activity in space. Essentially a 2d histogram weighted by
        neural activity.

        :parameters
        ---
        neuron: int, neuron index in neural_data.
        bin_size_cm: float, bin size in centimeters.
        show_plot: bool, flag for plotting.

        :return
        ---
        pf: (x,y) array, 2d histogram of position weighted by activity.
        """
        if shuffle:
            random_shift = randint(300, self.neural_data.shape[1])
            neural_data = np.roll(self.neural_data[neuron], random_shift)
        else:
            neural_data = self.neural_data[neuron]

        pf = spatial_bin(
            self.x,
            self.y,
            bin_size_cm=self.bin_size,
            show_plot=False,
            weights=neural_data,
            one_dim=self.one_dim,
        )[0]

        # Normalize by occupancy.
        if normalize_by_occ:
            pf = pf / self.occupancy_map

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(pf, origin="lower")

        return pf


# Adrien Peyrache's formula.
# def spatial_information(tuning_curve, occupancy):
#     tuning_curve = tuning_curve.flatten()
#     occupancy = occupancy.flatten()
#
#     occupancy = occupancy/np.sum(occupancy)
#     f = np.sum(occupancy*tuning_curve)

#     # This line is Matlab code. How do you do this in Python?
#     # Note: not element-wise division.
#     tuning_curve = tuning_curve/f

#     idx = tuning_curve > 0
#     SB = (occupancy[idx]*tuning_curve[idx])*np.log2(tuning_curve[idx])
#     SpatialBits = np.sum(SB)
#
#     return SpatialBits


def spatial_information(tuning_curve, occupancy):
    """
    Calculate spatital information in one neuron's activity.

    :parameters
    ---
    tuning_curve: array-like
        Activity (S or binary S) per spatial bin.

    occupancy: array-like
        Time spent in each spatial bin.

    :return
    ---
    spatial_bits_per_spike: float
        Spatial bits per spike.
    """
    # Make 1-D.
    tuning_curve = tuning_curve.flatten()
    occupancy = occupancy.flatten()

    # Only consider activity in visited spatial bins.
    tuning_curve = tuning_curve[occupancy > 0]
    occupancy = occupancy[occupancy > 0]

    # Find rate and mean rate.
    rate = tuning_curve / occupancy
    mrate = tuning_curve.sum() / occupancy.sum()

    # Get occupancy probability.
    prob = occupancy / occupancy.sum()

    # Handle log2(0).
    index = rate > 0

    # Spatial information formula.
    bits_per_spk = sum(
        prob[index] * (rate[index] / mrate) * np.log2(rate[index] / mrate)
    )

    return bits_per_spk


if __name__ == "__main__":
    mouse = "G132"
