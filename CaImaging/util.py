import os
from natsort import natsorted
import glob
import pandas as pd
import numpy as np
import cv2
import itertools
from csv import DictReader
import math
import matplotlib.pyplot as plt
import tkinter as tk

from CaImaging.Miniscope import open_minian

tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog


def concat_avis(path=None, pattern='behavCam*.avi',
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
    if path is None:
        path = filedialog.askdirectory()
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
                'distance': np.asarray(df['Distance_px'])} # Distance traveled since last sample

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



def consecutive_dist(x, axis=0, zero_pad=False):
    """
    Calculates the the distance between consecutive points in a vector.

    :parameter
    ---
    x: (N,2) array
        X and Y coordinates

    :return
    ---
    dists: array-like, distances.
    """
    delta = np.diff(x, axis=axis)
    dists = np.hypot(delta[:,0], delta[:,1])

    if zero_pad:
        dists = np.insert(dists, 0, 0)

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


def find_closest(array, value, sorted=False):
    """
    Finds the closest value in an array to a specified input.

    :parameters
    ---
    array: array-like
        Array to search.

    value: scalar
        Any value.

    :returns
    ---
    idx: int
        Index of array that is closest to value.

    array[idx]: anything
        Closest value in the array.
    """
    if sorted:
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1, array[idx - 1]
        else:
            return idx, array[idx]
    else:
        idx = (np.abs(array - value)).argmin()

        return idx, array[idx]


def get_data_paths(session_folder, pattern_dict):
    paths = {}
    for type, pattern in pattern_dict.items():
        match = glob.glob(os.path.join(session_folder, pattern))
        assert len(match) < 2, (f'Multiple possible files/folders detected. '
                                f'{match}')

        try:
            paths[type] = match[0]
        except:
            print(type + ' not found.')
            paths[type] = None

    return paths


class ScrollPlot:
    def __init__(self,
                 plot_function,
                 nrows=1,
                 ncols=1,
                 titles=None,
                 figsize=(8, 6),
                 current_position=0,
                 vid_fpath=None,
                 **kwargs):
        """
        Allows you to plot basically anything iterative and scroll through it.

        :parameters
        ---
        plot_function: function
            This function should be written specifically to draw from attrs
            in this class and plot them.

        nrows, ncols: ints
            Number of rows or columns in subplots.

        titles: list of strs
            Should be the same length as the data you want to plot.

        figsize: tuple
            Size of figure.

        current_position: int
            Defines the starting point for plotting your iterative stuff.

        kwargs: any key, value combination
            This is where data to be plotted would go. Also handles any
            other misc data.
        """
        self.plot_function = plot_function
        self.nrows = nrows
        self.ncols = ncols
        self.titles = titles
        self.figsize = figsize
        self.current_position = current_position
        for key, value in kwargs.items():
            setattr(self, key, value)

        # In cases where you need to read a video, do so during init.
        # current_position will set the starting frame.
        if vid_fpath is not None:
            self.vid = cv2.VideoCapture(vid_fpath, cv2.CAP_FFMPEG)
            self.vid.set(1, self.current_position)

        # Make figure.
        self.fig, (self.ax) = plt.subplots(self.nrows,
                                           self.ncols,
                                           figsize=self.figsize)

        # Plot then apply title.
        self.plot_function(self)
        try:
            self.ax.set_title(self.titles[self.current_position])
        except:
            pass

        # Connect to keyboard.
        self.fig.canvas.mpl_connect('key_press_event',
                                    lambda event: self.update_plots(event))


    def scroll(self, event):
        """
        Scrolls backwards or forwards using arrow keys. Quit with Esc.
        """
        if event.key == 'right' and self.current_position < self.last_position:
            self.current_position += 1
        elif event.key == 'left' and self.current_position > 0:
            self.current_position -= 1
        elif event.key == 'escape':
            plt.close(self.fig)

    def update_plots(self, event):
        """
        Update the plot with new index.
        """
        try:
            for ax in self.fig.axes:
                ax.cla()
        except:
            self.ax.cla()

        # Scroll then update plot.
        self.scroll(event)
        self.plot_function(self)
        try:
            self.ax.set_title(self.titles[self.current_position])
        except:
            pass
        self.fig.canvas.draw()


def disp_frame(ScrollObj):
    """
    Display frame and tracked position. Must specify path to video file
    during ScrollPlot class instantiation as well as positional data.
    To use:
        f = ScrollPlot(disp_frame, current_position = 0,
                       vid_fpath = 'path_to_video',
                       x = x_position, y = y_position,
                       titles = frame_numbers)

    """
    # Check that ScrollPlot object has all these attrs.
    attrs = ['vid', 'x', 'y']
    for attr in attrs:
        assert hasattr(ScrollObj, attr), (attr + ' missing')

    # Read the frame.
    try:
        ScrollObj.vid.set(1, ScrollObj.current_position)
        ret, frame = ScrollObj.vid.read()
    except:
        raise ValueError('Something went wrong with reading video.')

    # Plot the frame and position.
    ScrollObj.ax.imshow(frame)
    ScrollObj.ax.scatter(ScrollObj.x[ScrollObj.current_position],
                         ScrollObj.y[ScrollObj.current_position],
                         marker='+', s=80, c='r')

    # Find limit.
    ScrollObj.last_position = int(ScrollObj.vid.get(7)) - 1


def sync_cameras(timestamp_fpath, miniscope_cam=6, behav_cam=2):
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp file from miniscope recordings.

    Parameters
    ----------
    timestamp_fpath: str
        Full file path to timestamp.dat file.
        Yields timestamp dataframe. should contain field 'frameNum', 'camNum' and 'sysClock'

    miniscope_cam: int
        Number corresponding to miniscope camera.

    behav_cam: int
        Number corresponding to behavior camera.

    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
        fmCam0 is the miniscope cam.
        fmCam1 is the behavior cam.
    """
    ts = pd.read_csv(timestamp_fpath, sep="\s+")
    #cam_change = behav_cam - miniscope_cam

    # ts["change_point"] = ts["camNum"].diff()
    ts["ts_behav"] = np.where(ts["camNum"] == behav_cam,
                              ts["sysClock"], np.nan)
    ts["ts_forward"] = ts["ts_behav"].fillna(method="ffill")
    ts["ts_backward"] = ts["ts_behav"].fillna(method="bfill")
    ts["diff_forward"] = np.absolute(ts["sysClock"] - ts["ts_forward"])
    ts["diff_backward"] = np.absolute(ts["sysClock"] - ts["ts_backward"])
    ts["fm_behav"] = np.where(ts["camNum"] == behav_cam,
                              ts["frameNum"], np.nan)
    ts["fm_forward"] = ts["fm_behav"].fillna(method="ffill")
    ts["fm_backward"] = ts["fm_behav"].fillna(method="bfill")
    ts["fmCam1"] = np.where(
        ts["diff_forward"] < ts["diff_backward"], ts["fm_forward"], ts["fm_backward"]
    )
    ts_map = (
        ts[ts["camNum"] == miniscope_cam][["frameNum", "fmCam1"]]
            .dropna()
            .rename(columns=dict(frameNum="fmCam0"))
            .astype(dict(fmCam1=int))
    )
    ts_map["fmCam0"] = ts_map["fmCam0"] - 1
    ts_map["fmCam1"] = ts_map["fmCam1"] - 1

    return ts_map, ts


def sync_data(csv_path, minian_path, timestamp_path,
              miniscope_cam=6, behav_cam=1):
    """
    Synchronizes minian and behavior time series.

    :parameters
    ---
    csv_path: str
        Full file path to a csv from ezTrack.

    minian_path: str
        Full file path to the folder containing the minian folder.

    timestamp_path: str
        Full file path to the timestamp.dat file.

    miniscope_cam: int
        Camera number corresponding to the miniscope.

    behav_cam: int
        Camera nubmer corresponding to the behavior camera.

    :return
    ---
    synced_behavior: DataFrame
        The behavior csv, except downsampled and reordered
        according to the frames present in minian.

    """
    # Open behavior csv (from ezTrack) and minian files.
    behavior = pd.read_csv(csv_path)
    minian = open_minian(minian_path)

    # Match up timestamps from minian and behavior. Use minian
    # as the reference.
    ts_map, ts = sync_cameras(timestamp_path,
                              miniscope_cam=miniscope_cam,
                              behav_cam=behav_cam)
    miniscope_frames = np.asarray(minian.C.frame)
    behavior_frames = ts_map.fmCam1.iloc[miniscope_frames]

    # Rearrange all the behavior frames.
    synced_behavior = behavior.iloc[behavior_frames]

    return synced_behavior



def nan_array(size):
    arr = np.empty(size)
    arr.fill(np.nan)

    return arr


if __name__ == '__main__':
    folder = r'Z:\Will\Susie\TS29-0\S10'
    minian_folder = r'Y:\Susie\2020\epilepsy_imaging_analysis\LTEEG\ByAnimal\TS29-0\S10'

    csv_path = os.path.join()