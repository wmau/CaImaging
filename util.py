import os
import xarray as xr
from moviepy.editor import VideoFileClip, concatenate_videoclips
from natsort import natsorted
import glob
import pandas as pd
import numpy as np
import cv2

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


def read_eztrack(path, cm_per_pixel=1):
    """
    Reads ezTrack outputs.

    Parameter
    ---
    path: str, path to folder containing a csv from ezTrack.
    cm_per_pixel: float, centimeters per pixel.

    Return
    ---
    position: dict, with keys x, y, frame, distance.
    """
    # Open file.
    csv_fname = glob.glob(os.path.join(path, '*_tracked.csv'))
    assert len(csv_fname) is 1, "Warning: More than one csv found"
    df = pd.read_csv(csv_fname[0])

    # Consolidate into dict.
    position = {'x': np.asarray(df['X']) * cm_per_pixel,    # x position
                'y': np.asarray(df['Y']) * cm_per_pixel,    # y position
                'frame': np.asarray(df['Frame']),           # Frame number
                'distance': np.asarray(df['Distance']) * cm_per_pixel} # Distance traveled since last sample

    return position


def synchronize_time_series(behavior, neural):
    """
    Synchronizes behavior and neural time series by interpolating behavior.

    """


if __name__ == '__main__':
    path = r'D:\Projects\GTime\Data\G123\2\H14_M46_S20'
    behav_path = os.path.join(path, 'Behavior')

    minian = open_minian(path)
    S = np.asarray(minian.S)

    position = read_eztrack(behav_path)

    synchronize_time_series(position['frame'], S)