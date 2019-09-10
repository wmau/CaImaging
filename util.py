import os
import xarray as xr
from moviepy.editor import VideoFileClip, concatenate_videoclips
from natsort import natsorted
import glob
import pandas as pd

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


def concat_avis(path, fname='Merged.avi', fps=20, codec='rawvideo'):
    """
    Concatenates behavioral avi files for ezTrack.

    Parameters
    ---
    path: str, path to folder containing avis. All avis will be merged.
    fname: str, file name to be saved as in the same directory as path.
    fps: float, sampling rate.
    codec: str, 'rawvideo' seems to work

    """
    L = []

    for root, dirs, files in os.walk(path):
        files = natsorted(files)    # Sort files.

        # Find all files and append to master list.
        for file in files:
            if os.path.splitext(file)[1] == '.avi':
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                L.append(video)

    # Concatenate and write.
    final_clip = concatenate_videoclips(L)
    final_clip_name = os.path.join(path, fname)
    final_clip.to_videofile(final_clip_name, fps=fps, codec=codec)

    return final_clip_name


def read_eztrack(path, cm_per_pixel=1):
    """
    Reads ezTrack outputs.

    Parameter
    ---
    path: str, path to folder containing a csv from ezTrack.
    """
    # Open file.
    csv_fname = glob.glob(os.path.join(path, '*_tracked.csv'))
    assert len(csv_fname) is 1, "Warning: More than one csv found"
    df = pd.read_csv(csv_fname[0])

    position = {'x': df['X'] * cm_per_pixel,
                'y': df['Y'] * cm_per_pixel,
                'frame': df['Frame'],
                'distance': df['Distance'] * cm_per_pixel}

    return position
