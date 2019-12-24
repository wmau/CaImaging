import os
import xarray as xr
from natsort import natsorted
import glob
import pandas as pd
import numpy as np
import cv2
import itertools
from csv import DictReader
from math import sqrt
from scipy.stats import norm

def open_minian(dpath, fname='minian', backend='zarr', chunks=None):
    """
    Opens minian outputs.

    Parameters
    ---
    dpath: str, path to folder containing the minian outputs folder.
    fname: str, name of the minian output folder.
    backend: str, 'zarr' or 'netcdf'. 'netcdf' seems outdated.
    chunks: ??
    """
    if backend is 'netcdf':
        fname = fname + '.nc'
        mpath = os.path.join(dpath, fname)
        with xr.open_dataset(mpath) as ds:
            dims = ds.dims
        chunks = dict([(d, 'auto') for d in dims])
        ds = xr.open_dataset(os.path.join(dpath, fname), chunks=chunks)

        return ds

    elif backend is 'zarr':
        mpath = os.path.join(dpath, fname)
        dslist = [xr.open_zarr(os.path.join(mpath, d))
                  for d in os.listdir(mpath)
                  if os.path.isdir(os.path.join(mpath, d))]
        ds = xr.merge(dslist)
        if chunks is 'auto':
            chunks = dict([(d, 'auto') for d in ds.dims])

        return ds.chunk(chunks)

    else:
        raise NotImplementedError("backend {} not supported".format(backend))


def concat_avis(path, pattern='behavCam*.avi',
                fname='Merged.avi', fps=30, isColor=True):
    """
    Concatenates behavioral avi files for ezTrack.

    Parameters
    ---
    path: str, path to folder containing avis. All avis will be merged.
    pattern: str, pattern of video clips.
    fname: str, file name of final merged clip.
    fps: int, sampling rate.
    isColor: bool, flag for writing color.

    Return
    ---
    final_clip_name: str, full file name of final clip.
    """
    # Get all files.
    files = natsorted(glob.glob(os.path.join(path, pattern)))

    # Get width and height.
    cap = cv2.VideoCapture(files[0])
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define writer.
    fourcc = 0
    final_clip_name = os.path.join(path, fname)
    writer = cv2.VideoWriter(final_clip_name, fourcc,
                             fps, size, isColor=isColor)

    for file in files:
        print(f'Processing {file}')
        cap = cv2.VideoCapture(file)
        cap.set(1,0)                # Go to frame 0.
        cap_max = int(cap.get(7))   #7 is the index for total frames.

        # Loop through all the frames.
        for frame_num in range(cap_max):
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale if specified.
                if not isColor:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                writer.write(frame)
            else:
                break

        cap.release()

    writer.release()
    print(f'Writing {final_clip_name}')

    return final_clip_name


def read_eztrack(csv_fname):
    """
    Reads ezTrack outputs.

    Parameters
    ---
    csv_fname: str, path to tracking .csv from ezTrack.
    cm_per_pixel: float, centimeters per pixel.

    Return
    ---
    position: dict, with keys x, y, frame, distance.
    """
    # Open file.
    df = pd.read_csv(csv_fname)

    # Consolidate into dict.
    position = {'x': np.asarray(df['X']),    # x position
                'y': np.asarray(df['Y']),    # y position
                'frame': np.asarray(df['Frame']),           # Frame number
                'distance': np.asarray(df['Distance_in'])} # Distance traveled since last sample

    return pd.DataFrame(position)


def synchronize_time_series(position, neural, behav_fps=30, neural_fps=15):
    """
    Synchronizes behavior and neural time series by interpolating behavior.

    :parameters
    ---
    position: dict, output from read_ezTrack().
    neural: (neuron, t) array, any time series output from minian (e.g., C, S).
    behav_fps: float, sampling rate of behavior video.
    neural_fps: float, sampling rate of minian data.

    :return
    ---
    position: dict, interpolated data based on neural sampling rate.

    """
    # Get number of frames in each video.
    neural_nframes = neural.shape[1]
    behav_nframes = len(position['frame'])

    # Create time vectors.
    neural_t = np.arange(0, neural_nframes/neural_fps, 1/neural_fps)
    behav_t = np.arange(0, behav_nframes/behav_fps, 1/behav_fps)

    # Interpolate.
    position['x'] = np.interp(neural_t, behav_t, position['x'])
    position['y'] = np.interp(neural_t, behav_t, position['y'])
    position['frame'] = np.interp(neural_t, behav_t, position['frame'])

    # Normalize.
    position['x'] = position['x'] - min(position['x'])
    position['y'] = position['y'] - min(position['y'])

    # Compute distance at each consecutive point.
    pos_diff = np.diff(position['x']), np.diff(position['y'])
    position['distance'] = np.hypot(pos_diff[0], pos_diff[1])

    # Compute velocity by dividing by 1/fps.
    position['velocity'] = \
        np.concatenate(([0], position['distance']*min((neural_fps, behav_fps))))

    return position


def get_transient_timestamps(neural_data, std_thresh=3):
    """
    Converts an array of continuous time series (e.g., traces or S)
    into lists of timestamps where activity exceeds some threshold.

    :parameters
    ---
    neural_data: (neuron, time) array
        Neural time series, (e.g., C or S).

    std_thresh: float
        Number of standard deviations above the mean to define threshold.

    :returns
    ---
    event_times: list of length neuron
        Each entry in the list contains the timestamps of a neuron's
        activity.

    event_mags: list of length neuron
        Event magnitudes.

    """
    # Compute thresholds for each neuron.
    stds = np.std(neural_data, axis=1)
    means = np.mean(neural_data, axis=1)
    thresh = means + std_thresh*stds

    # Get event times and magnitudes.
    bool_arr = neural_data > np.tile(thresh,[neural_data.shape[1], 1]).T

    event_times = [np.where(neuron > t)[0] for neuron, t
                   in zip(neural_data, thresh)]

    event_mags = [neuron[neuron > t] for neuron, t
                  in zip(neural_data, thresh)]

    return event_times, event_mags, bool_arr


def distinct_colors(n):
    """
    Returns n colors that are maximally psychophysically distinct.

    :parameter
    ---
    n: int
        Number of colors that you want.

    :return
    colors: n list of strings
        Each str is a hex code corresponding to a color.

    """
    def MidSort(lst):
        if len(lst) <= 1:
            return lst
        i = int(len(lst) / 2)
        ret = [lst.pop(i)]
        left = MidSort(lst[0:i])
        right = MidSort(lst[i:])
        interleaved = [item for items in itertools.zip_longest(left, right)
                       for item in items if item != None]
        ret.extend(interleaved)
        return ret


    # Build list of points on a line (0 to 255) to use as color 'ticks'
    max_ = 255
    segs = int(n ** (1 / 3))
    step = int(max_ / segs)
    p = [(i * step) for i in np.arange(1, segs)]
    points = [0, max_]
    points.extend(MidSort(p))

    # Not efficient!!! Iterate over higher valued 'ticks' first (the points
    #   at the front of the list) to vary all colors and not focus on one channel.
    colors = ["#%02X%02X%02X" % (points[0], points[0], points[0])]
    r = 0
    total = 1
    while total < n and r < len(points):
        r += 1
        for c0 in range(r):
            for c1 in range(r):
                for c2 in range(r):
                    if total >= n:
                        break
                    c = "#%02X%02X%02X" % (points[c0], points[c1], points[c2])
                    if c not in colors and c != '#FFFFFF':
                        colors.append(c)
                        total += 1

    return colors


def ordered_unique(sequence):
    """
    Returns unique values in sequence that's in the same order as
    presented.

    :parameter
    ---
    sequence: array-like
        Sequence of anything.

    :return
    ---
    unique_items: list
        Contains unique items from list, in order of presentation.

    """
    seen = set()
    seen_add = seen.add

    return [x for x in sequence if not (x in seen or seen_add(x))]


def dir_dict(csv_path=r'D:\Projects\GTime\Data\GTime1.csv'):
    """
    Converts a csv containing session metadata into a dict. Useful
    for compiling all the paths for a project. csv should have
    some number of columns with a label.

    :parameter
    ---
    csv_path: str
        Path to csv. Default is the GTime1.csv on Will's desktop.

    :return
    ---
    dict_list: list of dicts
        List of dictionaries corresponding to the number of rows in
        the csv. Each entry in the list has keys (columns) filled in
        with the value for each row in the csv.

    """
    dict_list = []
    with open(csv_path, 'r') as file:
        csv_file = DictReader(file)

        for entry in csv_file:
            dict_list.append({k: v for k, v in zip(entry.keys(),
                                                   entry.values())})

    return dict_list


def find_dict_entries(dict_list, mode='and', **kwargs):
    """
    Finds the dictionary entry (or entries) that corresponds to
    the key and value in kwargs

    To get all AM entries in G132:
    find_dict_entries(dict_list, mode='and', **{'Animal: 'G132', 'Notes':'AM'})

    Replacing the above mode with 'or' gets all AM entries
    or all G132 entries.

    :parameters
    ---
    dict_list: list of dicts
        From dir_dict()

    mode: str ('and', 'or')
        Whether to intersect or union the entries that match
        the kwargs dict.

    kwargs: dict with any number of keys, values
        The function will match dict_list entries to these pairings.

    :return
    ---
    entries: list of dicts
        Entries that satisfy key=value.
    """
    if mode == 'and':
        for key, value in kwargs.items():
            dict_list = [d for d in dict_list if d[key] == value]

        entries = dict_list
    elif mode == 'or':
        entries = []
        for key, value in kwargs.items():
            entries.extend(d for d in dict_list if d[key] == value and d not in entries)
    else:
        TypeError('Mode not supported')

    return entries


def load_session(master_csv=r'D:\Projects\GTime\Data\GTime1.csv',
                 data_key='DataPath', behavior_key='BehaviorCSV',
                 minian_attr='S', **kwargs):
    """
    Loads and synchronizes behavior and imaging. Specify which
    sessions to load in the kwargs dict (which is fed into find_dict_entries).

    :parameters
    ---
    master_csv: str
        Path to the master csv that contains metadata for all sessions.

    data_key: str
        The dict key (csv column title) corresponding to minian path.

    behavior_key: str
        Dict key corresponding to behavior CSV.

    minian_attr: str
        Data from minian output that you want to load in.

    kwargs: dict
        Key, value pairings specifying which sessions you want to load.
        Example: {'Animal': 'G132', 'Notes': 'AM'}

    """
    # Get sessions.
    dict_list = dir_dict(master_csv)
    sessions = find_dict_entries(dict_list, **kwargs)

    for session in sessions:
        # Load neural data.
        session['NeuralData'] = \
            np.asarray(getattr(open_minian(session[data_key]),
                               minian_attr))

        # Load and sync behavior.
        session['Behavior'] = \
            synchronize_time_series(read_eztrack(session[behavior_key]),
                                    session['NeuralData'])

    return sessions


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def consecutive_dist(x, axis=0):
    """
    Calculates the the distance between consecutive points in a vector.

    :parameter
    ---
    x: array-like, vector of values.

    :return
    ---
    dists: array-like, distances.
    """
    delta = np.diff(x, axis=axis)
    dists = np.hypot(delta[:,0], delta[:,1])

    return dists


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def find_closest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx, array[idx]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    load_session(**{'Animal': 'G132'})