from natsort import natsorted
import os
import dask.array as darr
import xarray as xr
import numpy as np
import warnings
import dask as da
import cv2
import re
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt
import pandas as pd


def load_videos(
    vpath,
    pattern="msCam[0-9]+\.avi$",
    dtype=np.float64,
    downsample=None,
    downsample_strategy="subset",
    post_process=None,
):
    """Load videos from a folder.
    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together across time and return a
    `xarray.DataArray` representation of the concatenated videos. The default
    assumption is video filenames start with ``msCam`` followed by at least a
    number, and then followed by ``.avi``. In addition, it is assumed that the
    name of the folder correspond to a recording session identifier.
    Parameters
    ----------
    vpath : str
        The path to search for videos
    pattern : str, optional
        The pattern that describes filenames of videos. (Default value =
        'msCam[0-9]+\.avi')
    Returns
    -------
    xarray.DataArray or None
        The labeled 3-d array representation of the videos with dimensions:
        ``frame``, ``height`` and ``width``. Returns ``None`` if no data was
        found in the specified folder.
    """
    vpath = os.path.normpath(vpath)
    ssname = os.path.basename(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension == ".avi":
        movie_load_func = load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = load_tif_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        bin_eg = {d: np.arange(0, varr.sizes[d], w) for d, w in downsample.items()}
        if downsample_strategy == "mean":
            varr = (
                varr.coarsen(**downsample, boundary="trim")
                .mean()
                .assign_coords(**bin_eg)
            )
        elif downsample_strategy == "subset":
            varr = varr.sel(**bin_eg)
        else:
            warnings.warn("unrecognized downsampling strategy", RuntimeWarning)
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, ssname, vlist, varr_list)
    return varr


def load_avi_lazy(fname):
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_avi_perframe(fname, fid):
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print("frame read failed for frame {}".format(fid))
        return np.zeros((h, w))


def load_tif_lazy(fname):
    with TiffFile(fname) as tif:
        data = tif.asarray()

    f = int(data.shape[0])
    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_tif_perframe(fname, fid):
    return imread(fname, key=fid)


def project_image(vpath, projection_type="min", fname="MinimumProjection.pdf"):

    data = open_minian(vpath)

    if projection_type == "min":
        proj = data.Y.min("frame").compute()
    elif projection_type == "max":
        proj = data.Y.max("frame").compute()

    fig, ax = plt.subplots()
    ax.imshow(proj, cmap="gray", origin="lower")

    fig_name = os.path.join(vpath, fname)
    fig.savefig(fig_name)

    return proj


def threshold_S(S, std_thresh=1):
    thresholded_S = np.asarray(
        [
            (activity > (np.mean(activity) + std_thresh * np.std(activity))).astype(int)
            for activity in S
        ],
        dtype=int,
    )

    return thresholded_S


def open_minian(dpath, fname="minian", backend="zarr", chunks=None):
    """
    Opens minian outputs.

    Parameters
    ---
    dpath: str, path to folder containing the minian outputs folder.
    fname: str, name of the minian output folder.
    backend: str, 'zarr' or 'netcdf'. 'netcdf' seems outdated.
    chunks: ??
    """
    if backend is "netcdf":
        fname = fname + ".nc"
        mpath = os.path.join(dpath, fname)
        with xr.open_dataset(mpath) as ds:
            dims = ds.dims
        chunks = dict([(d, "auto") for d in dims])
        ds = xr.open_dataset(os.path.join(dpath, fname), chunks=chunks)

        return ds

    elif backend is "zarr":
        mpath = os.path.join(dpath, fname)
        dslist = [
            xr.open_zarr(os.path.join(mpath, d))
            for d in os.listdir(mpath)
            if os.path.isdir(os.path.join(mpath, d))
        ]
        ds = xr.merge(dslist)
        if chunks is "auto":
            chunks = dict([(d, "auto") for d in ds.dims])

        return ds.chunk(chunks)

    else:
        raise NotImplementedError("backend {} not supported".format(backend))


if __name__ == "__main__":
    folder = r"C:\Users\wm228\OneDrive\Documents\Sinai\Projects\Circle track\M1\03_09_2020\H11_M38_S32"
    # data = sync_preprocessed_data(folder)

    pass
