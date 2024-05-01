from lightkurve import DataCube, DataFrame
from lightkurve.mixins import STATS_METHOD_NAMES
import numpy as np
import pandas as pd

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
assert isinstance(df[:, row, col], DataFrame)
assert df[:, row, col].ntime == ntime
assert df[:, row, col].shape == (ntime, 9)

# timeseries
assert isinstance(df[:, [1, 2, 3], [1, 2, 3]], DataFrame)
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
