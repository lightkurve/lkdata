from lightkurve import DataCube, ErrorCube, DataFrame, ErrorFrame, DataSeries, TESTDATA
from lightkurve.mixins import STATS_METHOD_NAMES
import numpy as np
import pandas as pd
from astropy.io import fits

# Example usage
ntime, nrow, ncol = 200, 10, 14
test_data = np.random.normal(size=(ntime, nrow, ncol))
df = DataCube(test_data)
# This aperture makes it timeseries
aperture = np.zeros((10, 11), bool)
aperture[1:4, 1:4] = True

assert df.ntime == ntime
assert df.nrow == nrow
assert df.ncol == ncol

assert df.to_array().shape == (ntime, nrow, ncol)
assert np.allclose(df.to_array(), test_data)

# Should this be an image?
assert isinstance(df[0], DataCube)
assert df[0].ntime == 1
assert df[0].nrow == nrow
assert df[0].ncol == ncol

# This is still a 3D frame
assert isinstance(df[:, :2], DataCube)
assert df[:, :2].ntime == ntime
assert df[:, :2].nrow == 2
assert df[:, :2].ncol == ncol

assert isinstance(df[:, :, :2], DataCube)
assert df[:, :, :2].ntime == ntime
assert df[:, :, :2].nrow == nrow
assert df[:, :, :2].ncol == 2


assert isinstance(df[:, aperture], pd.DataFrame)
assert df[:, aperture].shape == (ntime, 9)

# This should be a timeseries
row, col = np.where(aperture)
assert isinstance(df[:, row, col], DataSeries)
assert df[:, row, col].ntime == ntime
assert df[:, row, col].shape == (ntime, 9)

# timeseries
assert isinstance(df[:, [1, 2, 3], [1, 2, 3]], DataSeries)
assert df[:, [1, 2, 3], [1, 2, 3]].ntime == ntime
assert df[:, [1, 2, 3], [1, 2, 3]].shape == (ntime, 3)

# DataCube
assert isinstance(df[:, 0, :], DataFrame)
assert df[:, 0, :].ntime == ntime
assert df[:, 0, :].shape == (ntime, ncol)

# DataCube
assert isinstance(df[:, :, 0], DataFrame)
assert df[:, :, 0].ntime == ntime
assert df[:, :, 0].shape == (ntime, nrow)

# DataCube
assert isinstance(df[:, 1, 0], DataFrame)
assert df[:, 1, 0].ntime == ntime
assert df[:, 1, 0].shape == (ntime, 1)

# TimeSeries
assert isinstance(df[:, :1, [1, 2]], DataFrame)
assert df[:, :1, [1, 2]].ntime == ntime
assert df[:, :1, [1, 2]].shape == (ntime, 2)

# TimeSeries
assert isinstance(df[:, [0, 1], [1, 2]], DataFrame)
assert df[:, [0, 1], [1, 2]].ntime == ntime
assert df[:, [0, 1], [1, 2]].shape == (ntime, 2)

for method_name in STATS_METHOD_NAMES:
    assert getattr(df, method_name)(axis=0).shape == (nrow, ncol)
    assert getattr(df[:, aperture], method_name)(axis=0).shape[0] == aperture.sum()

assert (df.mean() == df.mean(axis=0)).all()

# Example usage
ntime, nrow, ncol = 200, 10, 14
test_data = np.ones((ntime, nrow, ncol))
test_data = np.ones((ntime, nrow, ncol))
row, col = np.arange(test_data.shape[1]), np.arange(test_data.shape[2])
df = DataCube(test_data)
# Time downsampling
assert (df.downsample(2).to_array() == 2).all()
assert df.spatial_downsample(2).to_array.shape == (200, 5, 7)
assert (df.spatial_downsample(2).to_array() == 4).all()
assert (df.spatial_aggregate(5, 7).to_array().round() == 4).all()
assert df[:, :, :-1].spatial_downsample(2).to_array().shape == (200, 5, 6)
assert (
    df[:, :, :-1].spatial_downsample(2).to_array()
    == df[:, :, :-2].spatial_downsample(2).to_array
).all()
assert df[:, :-1, :].spatial_downsample(2).to_array().shape == (200, 4, 7)


def make_test_data():
    hdulist = fits.open(TESTDATA)
    flux_array = hdulist[1].data["FLUX"].astype(float)
    flux_err_array = hdulist[1].data["FLUX_ERR"].astype(float)
    time = hdulist[1].data["TIME"].astype(float)
    time_corr = hdulist[1].data["TIMECORR"].astype(float)
    c0, r0 = hdulist[1].header["1CRV4P"], hdulist[1].header["2CRV4P"]
    row, col = np.arange(flux_array.shape[1]) + r0, np.arange(flux_array.shape[2]) + c0
    aper = flux_array.mean(axis=0) > 10000
    bkg_aper = flux_array.mean(axis=0) < 4000

    time_mask = hdulist[1].data["QUALITY"] == 0

    flux = DataCube(
        flux_array,
        time_indices={"btjd": time, "spacecraft_time": time - time_corr},
        row_indices={"row": row},
        col_indices={"column": col},
    )

    flux_err = ErrorCube(
        flux_err_array,
        time_indices={"btjd": time, "spacecraft_time": time - time_corr},
        row_indices={"row": row},
        col_indices={"column": col},
    )
    return flux, flux_err, aper, bkg_aper, time_mask


flux, flux_err, aper, bkg_aper, time_mask = make_test_data()

assert isinstance(flux, DataCube)
assert isinstance(flux_err, ErrorCube)

assert flux.to_array().shape == (50, 6, 6)
assert flux_err.to_array().shape == (50, 6, 6)

assert flux.downsample(5).to_array().shape == (8, 6, 6)
assert flux_err.downsample(5).to_array().shape == (8, 6, 6)

assert isinstance(flux[:, aper], DataFrame)
assert isinstance(flux_err[:, aper], ErrorFrame)

assert isinstance(flux[:, aper].sum(axis=1), DataSeries)
assert isinstance(flux_err[:, aper].sum(axis=1), DataSeries)

assert flux.spatial_downsample(2).to_array().shape == (50, 3, 3)
assert flux.spatial_downsample(2).to_array()[0, 0, 0] == flux[0, :2, :2].sum().sum()
assert flux.spatial_downsample(2).to_array()[0, -1, -1] == flux[0, -2:, -2:].sum().sum()
assert flux.spatial_downsample(2).sum().sum() == flux.sum().sum()
assert flux[:, :-1].spatial_downsample(2).sum().sum() == flux[:, :-2].sum().sum()
assert flux[:, :, :-1].spatial_downsample(2).sum().sum() == flux[:, :, :-2].sum().sum()
