import numpy as np
from CaImaging.Behavior import spatial_bin
import matplotlib.pyplot as plt
from random import randint
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
from CaImaging.util import consecutive_dist
from scipy.signal import savgol_filter
from sklearn.impute import SimpleImputer

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 12})


class PlaceFields:
    def __init__(
        self,
        t,
        x,
        y,
        neural_data,
        bin_size=20,
        circular=False,
        shuffle_test=False,
        fps=None,
        velocity_threshold=10,
        circle_radius=1,
    ):
        """
        Place field object.

        :parameters
        ---
        t: array
            Time array in milliseconds.

        x, y: (t,) arrays
            Positions per sample. Should be in cm. If circular==True,
            x can also be in radians, but you should also use
            circle_radius. And also if circular==True, y should be
            an array of zeros.

        neural_data: (n,t) array
            Neural activity (usually S).

        bin_size: int
            Bin size in cm.

        circular: bool
            Whether the x data is in radians (for circular tracks).

        shuffle_test: bool
            Flag to shuffle data in time to recompute spatial information.

        fps: int
            Sampling rate. If None, will try to compute based on supplied
            time vector.

        velocity_threshold: float
            Velocity to threshold whether animal is running or not.

        circle_radius: float
            Radius of circular track in cm (38.1 for Will's tracK).


        """
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        neural_data = imp.fit_transform(neural_data.T).T

        self.data = {
            "t": t,
            "x": x,
            "y": y,
            "neural": neural_data,
            "n_neurons": neural_data.shape[0],
        }
        self.meta = {
            "circular": circular,
            "bin_size": bin_size,
            "velocity_threshold": velocity_threshold,
        }

        if self.meta["circular"]:
            self.meta["circle_radius"] = circle_radius

        # If we're using circular position, data must be in radians.
        if any(self.data["x"] > 2 * np.pi) and self.meta["circular"]:
            raise ValueError("x must be [0, 2pi]")

        # Get fps.
        if fps is None:
            self.meta["fps"] = self.get_fps()
        else:
            self.meta["fps"] = int(fps)

        # Compute distance and velocity. Smooth the velocity.
        d = consecutive_dist(
            np.asarray((self.data["x"], self.data["x"])).T, zero_pad=True
        )
        if self.meta["circular"]:
            too_far = d > np.pi
            d[too_far] = abs((2 * np.pi) - d[too_far])
        self.data["velocity"] = d / (1 / self.meta["fps"])

        # Convert radians to arc length by the formula s=r*theta.
        if self.meta["circular"]:
            self.data["velocity"] *= circle_radius
        self.data["velocity"] = savgol_filter(
            self.data["velocity"], self.meta["fps"], 1
        )
        self.data["running"] = self.data["velocity"] > self.meta["velocity_threshold"]

        # Make occupancy maps and place fields.
        (
            self.data["occupancy_map"],
            self.data["occupancy_bins"],
        ) = self.make_occupancy_map(show_plot=False)
        self.data["placefields"] = self.make_all_place_fields()
        self.data["spatial_info"] = [
            spatial_information(pf, self.data["occupancy_map"])
            for pf in self.data["placefields"]
        ]
        self.data["placefield_centers"] = self.find_pf_centers()
        if shuffle_test:
            (
                self.data["spatial_info_pvals"],
                self.data["spatial_info_z"],
            ) = self.assess_spatial_sig_parallel()

    def get_fps(self):
        """
        Get sampling frequency by counting interframe interval.

        :return:
        """
        # Take difference.
        interframe_intervals = np.diff(self.data["t"])

        # Inter-frame interval in milliseconds.
        mean_interval = np.mean(interframe_intervals)
        fps = round(1 / (mean_interval / 1000))

        return int(fps)

    def make_all_place_fields(self):
        """
        Compute the spatial rate maps of all neurons.

        :return:
        """
        pfs = []
        for neuron in range(self.data["neural"].shape[0]):
            pfs.append(self.make_place_field(neuron, show_plot=False))

        return np.asarray(pfs)

    def make_snake_plot(self, order="sorted", neurons="all", normalize=True):
        if neurons == "all":
            neurons = np.asarray([int(n) for n in range(self.data["n_neurons"])])
        pfs = self.data["placefields"][neurons]

        if order == "sorted":
            order = np.argsort(self.data["placefield_centers"][neurons])

        if normalize:
            pfs = pfs / pfs.nanmax(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        ax.imshow(pfs[order])

    def find_pf_centers(self):
        centers = [np.argmax(pf) for pf in self.data["placefields"]]

        return np.asarray(centers)

    def assess_spatial_sig(self, neuron, n_shuffles=500):
        shuffled_SIs = []
        for i in range(n_shuffles):
            shuffled_pf = self.make_place_field(neuron, show_plot=False, shuffle=True)
            shuffled_SIs.append(
                spatial_information(shuffled_pf, self.data["occupancy_map"])
            )

        shuffled_SIs = np.asarray(shuffled_SIs)
        p_value = np.sum(self.data["spatial_info"][neuron] < shuffled_SIs) / n_shuffles

        SI_z = (self.data["spatial_info"][neuron] - np.mean(shuffled_SIs)) / np.std(
            shuffled_SIs
        )

        return p_value, SI_z

    def assess_spatial_sig_parallel(self):
        print("Doing shuffle tests. This may take a while.")
        neurons = tqdm([n for n in range(self.data["n_neurons"])])
        n_cores = mp.cpu_count()
        # with futures.ProcessPoolExecutor() as pool:
        #     results = pool.map(self.assess_spatial_sig, neurons)
        results = Parallel(n_jobs=n_cores)(
            delayed(self.assess_spatial_sig)(i) for i in neurons
        )

        pvals, SI_z = zip(*results)

        return pvals, SI_z

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
        thresh = np.mean(self.data["neural"][neuron]) + std_thresh * np.std(
            self.data["neural"][neuron]
        )
        supra_thresh = self.data["neural"][neuron] > thresh

        # Plot.
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.data["x"], self.data["y"], s=3, c=pos_color)
        ax.scatter(
            self.data["x"][supra_thresh],
            self.data["y"][supra_thresh],
            s=3,
            c=transient_color,
        )

    def make_occupancy_map(self, show_plot=True, ax=None):
        """
        Makes the occupancy heat cell_map of the animal.

        :parameters
        ---
        bin_size_cm: float, bin size in centimeters.
        show_plot: bool, flag for plotting.

        """
        temp = spatial_bin(
            self.data["x"],
            self.data["y"],
            bin_size_cm=self.meta["bin_size"],
            show_plot=show_plot,
            one_dim=self.meta["circular"],
        )
        occupancy_map, occupancy_bins = temp[0], temp[-1]

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.imshow(occupancy_map, origin="lower")

        return occupancy_map, occupancy_bins

    def make_place_field(
        self, neuron, show_plot=True, normalize_by_occ=True, ax=None, shuffle=False
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
            random_shift = randint(300, self.data["neural"].shape[1])
            neural_data = np.roll(self.data["neural"][neuron], random_shift)
        else:
            neural_data = self.data["neural"][neuron]

        pf = spatial_bin(
            self.data["x"][self.data["running"]],
            self.data["y"][self.data["running"]],
            bin_size_cm=self.meta["bin_size"],
            show_plot=False,
            weights=neural_data[self.data["running"]],
            one_dim=self.meta["circular"],
            bins=self.data["occupancy_bins"],
        )[0]

        # Normalize by occupancy.
        if normalize_by_occ:
            pf = pf / self.data["occupancy_map"]

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
