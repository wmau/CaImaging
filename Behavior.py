import cv2

from util import read_eztrack
import numpy as np
import os
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt


def make_tracking_video(vid_path, csv_path, output_fname='Tracking.avi',
                        start=0, stop=None, fps=30):
    """
    Makes a video to visualize licking at water ports and position of the animal.

    :parameters
    ---
    video_fname: str
        Full path to the behavior video.

    csv_fname: str
        Full path to the csv file from EZTrack.

    output_fname: str
        Desired file name for output. It will be saved to the same folder
        as the data.

    start: int
        Frame to start on.

    stop: int or None
        Frame to stop on or if None, the end of the movie.

    fps: int
        Sampling rate of the behavior camera.
    """
    # Get behavior video.
    vid = cv2.VideoCapture(vid_path)
    if stop is None:
        stop = int(vid.get(7))  # 7 is the index for total frames.

    # Save data to the same folder.
    folder = os.path.split(vid_path)[0]
    output_path = os.path.join(folder, output_fname)

    # Get EZtrack data.
    eztrack = read_eztrack(csv_path)

    # Make video.
    fig, ax = plt.subplots()
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, 100):
        for frame_number in np.arange(start, stop):
            # Plot frame.
            vid.set(1, frame_number)
            ret, frame = vid.read()
            ax.imshow(frame)

            # Plot position.
            x = eztrack.at[frame_number, 'x']
            y = eztrack.at[frame_number, 'y']
            ax.scatter(x, y, marker='+', s=60, c='r')

            ax.text(0, 0, 'Frame: ' + str(frame_number) +
                    '   Time: ' + str(np.round(frame_number/30, 1)) + ' s')

            ax.set_aspect('equal')
            plt.axis('off')

            writer.grab_frame()

            plt.cla()


def spatial_bin(x, y, bin_size_cm=20, plot=False, weights=None, ax=None):
    """
    Spatially bins the position data.

    :parameters
    ---
    x,y: array-like
        Vector of x and y positions in cm.

    bin_size_cm: float
        Bin size in centimeters.

    plot: bool
        Flag for plotting.

    weights: array-like
        Vector the same size as x and y, describing weights for
        spatial binning. Used for making place fields (weights
        are booleans indicating timestamps of activity).

    ax: Axis object
        If you want to reference an exis that already exists. If None,
        makes a new Axis object.

    :returns
    ---
    H: (nx,ny) array
        2d histogram of position.

    xedges, yedges: (n+1,) array
        Bin edges along each dimension.
    """
    # Calculate the min and max of position.
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]

    # Make bins.
    xbins = np.linspace(x_extrema[0], x_extrema[1],
                        np.round(np.diff(x_extrema) / bin_size_cm))
    ybins = np.linspace(y_extrema[0], y_extrema[1],
                        np.round(np.diff(y_extrema) / bin_size_cm))

    # Do the binning.
    H, xedges, yedges = np.histogram2d(y, x,
                                       [ybins, xbins],
                                       weights=weights)

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(H)

    return H, xedges, yedges

if __name__ == '__main__':
    vid_fname = r'D:\Projects\CircleTrack\Mouse1\12_20_2019\H14_M59_S12\Merged.avi'
    eztrack_fpath = r'D:\Projects\CircleTrack\Mouse1\12_20_2019\H14_M59_S12\Merged_LocationOutput.csv'
    make_tracking_video(vid_fname, eztrack_fpath)
    # Arduino_fpath = r'D:\Projects\CircleTrack\Mouse1\12_20_2019\H14_M59_S12.3896 1428.txt'
    # eztrack_fpath = r'D:\Projects\CircleTrack\Mouse1\12_20_2019\H14_M59_S12\Merged_LocationOutput.csv'
    #eztrack_data = sync_Arduino_outputs(Arduino_fpath, eztrack_fpath)[0]
    #clean_lick_detection(eztrack_data)
    #find_water_ports_circletrack(eztrack_fpath)