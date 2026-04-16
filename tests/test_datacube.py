import numpy as np
import pandas as pd
import pytest

from astropy.io import fits
from lkdata import (
    TESTDATA,
    DataCube,
    DataSeriesCollection,
    DataSeries,
    BoolCube,
)
from lkdata.utils.uncertainty import Uncertainty
from lkdata.mixins import STATS_METHOD_NAMES

ntime, nrow, ncol = 200, 10, 14
# Actual data values should be irrelevant for these tests
test_data = np.random.normal(size=(ntime, nrow, ncol))
df = DataCube(test_data)
# This aperture makes it timeseries
aperture = np.zeros((10, 14), bool)
aperture[1:4, 1:4] = True


def test_repr():
    """Test that the repr is as expected."""
    assert f"DataCube {df.array.shape}" in repr(df)


def test_setup():
    """Test basic attributes are assigned on init."""
    assert df.ntime == ntime
    assert df.nrow == nrow
    assert df.ncol == ncol

    assert df.array.shape == (ntime, nrow, ncol)
    assert np.allclose(df.array, test_data)

    # Test overridden pandas methods return correct shapes
    # Data products return tuples for and uncertainty
    methods = [
        method for method in STATS_METHOD_NAMES if method not in ["argmin", "argmax"]
    ]
    for method_name in methods:
        assert getattr(df, method_name)(axis=0)[0].shape == (nrow, ncol)
        assert (
            getattr(df[:, aperture], method_name)(axis=0)[0].shape[0] == aperture.sum()
        )

    # argmin and argmax should return a single 3D index for axis=None
    assert len(df.argmin()) == 3
    assert len(df.argmax()) == 3

    # argmin and argmax operate on axis=0 or 1 only
    assert df.argmin(axis=0).shape[0] == nrow * ncol
    assert df.argmax(axis=0).shape[0] == nrow * ncol
    assert df.argmin(axis=1).shape[0] == ntime
    assert df.argmax(axis=1).shape[0] == ntime

    with pytest.raises(np.exceptions.AxisError, match="axis 2 is out of bounds"):
        _ = df.argmin(axis=2)


def test_setup_index():
    df = DataCube(test_data, index=range(ntime, 2 * ntime))
    assert "given_index" in df.index.names


def test_bad_setup():
    # Actual data values should be irrelevant for these tests
    data_mismatch = np.random.normal(size=(ntime + 10, nrow, ncol))
    index = df.index
    columns = df.columns
    with pytest.raises(ValueError, match="Length of index"):
        _ = DataCube(data_mismatch, index=index, columns=columns)
    data_mismatch = np.random.normal(size=(ntime, nrow + 10, ncol))
    with pytest.raises(ValueError, match="Number of columns"):
        _ = DataCube(data_mismatch, index=index, columns=columns)


def test_reserved_names():
    """Make sure reserved names raise errors when used in the wrong places"""
    with pytest.raises(ValueError, match="Key 'row' is reserved"):
        df = DataCube(test_data, time_indices={"row": np.arange(200)})

    with pytest.raises(ValueError, match="Key 'col' is reserved"):
        df = DataCube(test_data, time_indices={"col": np.arange(200)})

    with pytest.raises(ValueError, match="Key 'time_index' is reserved"):
        df = DataCube(test_data, row_indices={"time_index": np.arange(10)})

    with pytest.raises(ValueError, match="Key 'time_index' is reserved"):
        df = DataCube(test_data, col_indices={"time_index": np.arange(10)})

    df = DataCube(
        test_data,
        time_indices={"time_index": np.arange(200)},
        row_indices={"row": np.arange(10)},
        col_indices={"col": np.arange(14)},
    )
    assert all(df.time_index == np.arange(200))
    # assert all(df.row == np.arange(10).tile)
    # assert all(df.col == np.arange(14).tile)


def test_slicing():
    """Test slicing a DataCube"""
    # Single time index, int
    # Should this be an image?
    assert isinstance(df[0], DataCube)
    assert df[0].ntime == 1
    assert df[0].nrow == nrow
    assert df[0].ncol == ncol

    # Giving tuple of slices, one for time, one for space
    # This is still a 3D frame
    assert isinstance(df[:, :2], DataCube)
    assert df[:, :2].ntime == ntime
    assert df[:, :2].nrow == 2
    assert df[:, :2].ncol == ncol

    # Slice time, row, and column
    assert isinstance(df[:, :2, :], DataCube)
    assert df[:, :2, :].ntime == ntime
    assert df[:, :2, :].nrow == 2
    assert df[:, :2, :].ncol == ncol

    assert isinstance(df[:, :, :2], DataCube)
    assert df[:, :, :2].ntime == ntime
    assert df[:, :, :2].nrow == nrow
    assert df[:, :, :2].ncol == 2

    assert isinstance(df[:, :2, :2], DataCube)
    assert df[:, :2, :2].ntime == ntime
    assert df[:, :2, :2].nrow == 2
    assert df[:, :2, :2].ncol == 2

    # Mixed slice and indices
    assert isinstance(df[:, [0, 1], :2], DataSeriesCollection)
    assert df[:, [0, 1], :2].ntime == ntime
    assert df[:, [0, 1], :2].nseries == 4

    assert all(df[:, [0, 1], 0] == df[:, [0, 1], [0, 0]])
    assert isinstance(df[:, [0, 1], 0], DataSeriesCollection)
    assert df[:, [0, 1], 0].ntime == ntime
    assert df[:, [0, 1], 0].nseries == 2

    # Slice for time, 2D aperture
    assert isinstance(df[:, aperture], pd.DataFrame)
    assert df[:, aperture].shape == (ntime, 9)

    # Frames - timeseries for multiple pixels
    row, col = np.where(aperture)
    assert isinstance(df[:, row, col], DataSeriesCollection)
    assert df[:, row, col].ntime == ntime
    assert df[:, row, col].shape == (ntime, 9)

    assert isinstance(df[:, [1, 2, 3], [1, 2, 3]], DataSeriesCollection)
    assert df[:, [1, 2, 3], [1, 2, 3]].ntime == ntime
    assert df[:, [1, 2, 3], [1, 2, 3]].shape == (ntime, 3)

    assert isinstance(df[:, 0, :], DataSeriesCollection)
    assert df[:, 0, :].ntime == ntime
    assert df[:, 0, :].shape == (ntime, ncol)

    assert isinstance(df[:, :, 0], DataSeriesCollection)
    assert df[:, :, 0].ntime == ntime
    assert df[:, :, 0].shape == (ntime, nrow)

    assert isinstance(df[:, :1, [1, 2]], DataSeriesCollection)
    assert df[:, :1, [1, 2]].ntime == ntime
    assert df[:, :1, [1, 2]].shape == (ntime, 2)

    assert isinstance(df[:, [0, 1], [1, 2]], DataSeriesCollection)
    assert df[:, [0, 1], [1, 2]].ntime == ntime
    assert df[:, [0, 1], [1, 2]].shape == (ntime, 2)

    # DataSeries - timeseries for single pixel
    assert isinstance(df[:, 1, 0], DataSeries)
    assert df[:, 1, 0].ntime == ntime
    assert df[:, 1, 0].shape == (ntime,)


ntime, nrow, ncol = 200, 10, 14  # avoid square data shapes
test_data = np.ones((ntime, nrow, ncol))
df = DataCube(test_data, uncertainty=test_data)


def test_downsample():
    """Test downsampling methods"""
    # Time downsample
    assert (df.downsample(2).dropna() == 2).all(axis=None)
    # Uncetainty adds in quadrature
    assert (
        df.downsample(4).uncertainty.array[
            ~np.isnan(df.downsample(4).uncertainty.array)
        ]
        == 2
    ).all()

    # Spatial downsample data
    assert df.spatial_downsample(2).array.shape == (200, 5, 7)
    assert df.spatial_downsample((2, 1)).array.shape == (200, 5, 14)
    assert df.spatial_downsample((1, 2)).array.shape == (200, 10, 7)

    # A spatial factor of 2 combines 4 pixels (a 2x2 grid)
    assert (df.spatial_downsample(2).array == 4).all()
    # Specify a different factor for row and column
    assert all(df.spatial_downsample((2, 1)) == 2)  # tuple input
    assert all(df.spatial_downsample(2, 1) == 2)  # args input
    assert all(df.spatial_downsample(row_factor=2, col_factor=1) == 2)  # kwargs input
    assert all(df.spatial_downsample((1, 2)) == 2)
    # Downsampling should drop data which don't have enough pixels in the bin
    assert df[:, :, :-1].spatial_downsample(2).array.shape == (200, 5, 6)
    assert (
        df[:, :, :-1].spatial_downsample(2).array
        == df[:, :, :-2].spatial_downsample(2).array
    ).all()
    assert df[:, :-1, :].spatial_downsample(2).array.shape == (200, 4, 7)

    # Spatial downsample uncertainty
    assert df.spatial_downsample(2).uncertainty.array.shape == (200, 5, 7)
    assert (df.spatial_downsample(2).uncertainty.array == 2).all()
    assert df[:, :, :-1].spatial_downsample(2).uncertainty.array.shape == (200, 5, 6)
    assert (
        df[:, :, :-1].spatial_downsample(2).uncertainty.array
        == df[:, :, :-2].spatial_downsample(2).uncertainty.array
    ).all()
    assert df[:, :-1, :].spatial_downsample(2).uncertainty.array.shape == (200, 4, 7)

    assert (df.spatial_aggregate(5, 7).array.round() == 4).all()
    assert (df.spatial_aggregate(5, 7).uncertainty.array.round() == 2).all()  #


def test_math():
    """Basic mathematical operations on data with uncertainty"""
    # addition
    df2 = df + df
    assert all(df2 == 2)
    assert (df2.uncertainty.array == np.sqrt(2)).all()

    df2 = df + 1
    assert all(df2 == 2)
    assert (df2.uncertainty.array == 1).all()

    # subtraction
    df2 = df - df
    assert all(df2 == 0)
    assert (df2.uncertainty.array == np.sqrt(2)).all()

    df2 = df - 1
    assert all(df2 == 0)
    assert (df2.uncertainty.array == 1).all()

    # multiplication
    df2 = df * 2
    assert all(df2 == 2)
    assert (df2.uncertainty.array == 2).all()

    df2 = df * df
    assert all(df2 == 1)
    assert (df2.uncertainty.array == np.sqrt(2)).all()

    df2 = DataCube(test_data * 2, uncertainty=test_data)
    df3 = df * df2
    assert all(df3 == 2)
    assert (df3.uncertainty.array == np.sqrt(5)).all()

    # division
    df2 = df / 2
    assert all(df2 == 1 / 2)
    assert (df2.uncertainty.array == 1 / 2).all()

    df2 = df / df
    assert all(df2 == 1)
    assert (df2.uncertainty.array == np.sqrt(2)).all()

    df2 = DataCube(test_data * 2, uncertainty=test_data)
    df3 = df2 / df
    assert all(df3 == 2)
    assert (df3.uncertainty.array == np.sqrt(5)).all()

    df3 = df / df2
    assert all(df3 == 1 / 2)
    assert (df3.uncertainty.array == np.sqrt(5) / 4).all()


def test_downsample_order():
    """Ensure that downsampling sorts on the level and isn't affected by order
    of operations.
    """
    t0 = np.linspace(0.1, 10, 99)
    t1 = np.append(t0[::3], [t0[1::3], t0[2::3]])
    range_repeated = np.repeat(range(1, 4), 33).reshape(99, 1, 1)
    cube = DataCube(
        range_repeated, uncertainty=range_repeated, time_indices={"t0": t0, "t1": t1}
    )
    assert all(
        cube.downsample(3, level="t1").sort_index(level="t1")
        == cube.sort_index(level="t1").downsample(3, level="t1")
    )

    ds = cube.downsample(nframes=33, level="t0")
    result = ds.uncertainty.array.flatten()
    assert all(ds.to_numpy().flatten() == np.array([i * 33 for i in range(1, 4)]))
    assert all(result == np.array([(i**2 * 33) ** 0.5 for i in range(1, 4)]))
    ds = cube.downsample(nframes=33, level="t1")
    result = ds.uncertainty.array.flatten()
    assert all(ds.to_numpy().flatten() == (1 + 2 + 3) * 11)
    assert all(result == ((1**2 + 2**2 + 3**2) * 11) ** 0.5)


def test_fold():
    time = np.arange(10, step=0.1)
    sine10 = np.sin(time)
    data = np.array([v * np.ones((10, 14)) for v in sine10]).reshape((100, 10, 14))
    time_indices = {"days": time, "hours": time / 24, "minutes": time / (24 * 60)}
    cube = DataCube(data)
    folded = cube.fold(2 * np.pi)
    assert "phase" in folded.index.names
    folded = cube.fold(2 * np.pi, t0=np.pi)
    assert "phase" in folded.index.names
    cube = DataCube(data, time_indices=time_indices)
    folded = cube.fold(2 * np.pi, level="days")
    assert "phase" in folded.index.names
    folded = folded.fold(2 * np.pi, label="phase")
    assert "phase" in folded.index.names
    cube.fold(2 * np.pi, level="days", inplace=True)
    assert "phase" in cube.index.names
    cube.sort_index(level="days", inplace=True)


def test_drop_level():
    time = np.arange(10, step=0.1)
    time_indices = {"days": time, "hours": time / 24, "minutes": time / (24 * 60)}

    sine10 = np.sin(time)
    data = np.array([v * np.ones((10, 14)) for v in sine10]).reshape((100, 10, 14))
    cube = DataCube(data, time_indices=time_indices)
    cube_dropped = cube.droplevel("minutes")
    assert "minutes" not in cube_dropped.index.names


def make_test_data():
    """Make a Cube from real data for testing."""
    with fits.open(TESTDATA) as hdulist:
        if hasattr(hdulist[1], "data"):
            flux_array = hdulist[1].data["FLUX"].astype(float)
            flux_err_array = hdulist[1].data["FLUX_ERR"].astype(float)
            time = hdulist[1].data["TIME"].astype(float)
            time_corr = hdulist[1].data["TIMECORR"].astype(float)
            c0, r0 = hdulist[1].header["1CRV4P"], hdulist[1].header["2CRV4P"]
            row, col = (
                np.arange(flux_array.shape[1]) + r0,
                np.arange(flux_array.shape[2]) + c0,
            )
            aper = flux_array.mean(axis=0) > 10000
            bkg_aper = flux_array.mean(axis=0) < 4000

            time_mask = hdulist[1].data["QUALITY"] == 0

            flux = DataCube(
                flux_array,
                uncertainty=flux_err_array,
                time_indices={"btjd": time, "spacecraft_time": time - time_corr},
                row_indices={"row": row},
                col_indices={"column": col},
            )

            return flux, aper, bkg_aper, time_mask
        else:
            raise AttributeError(f"HDUList from {TESTDATA} has no data.")


def test_real_data():
    """Test methods on real data"""
    flux, aper, _, _ = make_test_data()

    assert isinstance(flux, DataCube)
    assert isinstance(flux.uncertainty, Uncertainty)

    assert flux.array.shape == (50, 6, 6)
    assert flux.uncertainty.array.shape == (50, 6, 6)

    assert flux.downsample(5).array.shape == (10, 6, 6)
    assert flux.downsample(5).uncertainty.array.shape == (10, 6, 6)

    assert flux.downsample(5).dropna().shape == (8, 36)
    assert flux.downsample(5).uncertainty.array[
        ~np.isnan(flux.downsample(5).uncertainty.array[:, 0, 0])
    ].shape == (8, 6, 6)

    assert isinstance(flux[:, aper], DataSeriesCollection)
    assert isinstance(flux[:, aper].uncertainty, Uncertainty)

    assert isinstance(flux[:, aper].sum(axis=1), DataSeries)
    assert isinstance(flux[:, aper].sum(axis=1).uncertainty, Uncertainty)

    assert flux.spatial_downsample(2).array.shape == (50, 3, 3)
    assert flux.spatial_downsample(2).uncertainty.shape == (50, 3, 3)

    assert (
        flux.spatial_downsample(2).array[0, 0, 0] == flux[0, :2, :2].sum(axis=None)[0]
    )
    assert (
        flux.spatial_downsample(2).uncertainty.array[0, 0, 0]
        == ((flux[0, :2, :2].uncertainty.array ** 2).sum().sum()) ** 0.5
    )

    assert (
        flux.spatial_downsample(2).array[0, -1, -1]
        == flux[0, -2:, -2:].sum(axis=None)[0]
    )
    assert (
        flux.spatial_downsample(2).uncertainty.array[0, -1, -1]
        == ((flux[0, -2:, -2:].uncertainty.array ** 2).sum(axis=None)) ** 0.5
    )

    assert flux.spatial_downsample(2).sum(axis=None)[0] == flux.sum(axis=None)[0]
    assert (flux.spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux.uncertainty.array**2).sum(axis=None).round()

    assert (
        flux[:, :-1].spatial_downsample(2).sum(axis=None)[0]
        == flux[:, :-2].sum(axis=None)[0]
    )
    assert (flux[:, :-1].spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux[:, :-2].uncertainty.array ** 2).sum(axis=None).round()
    assert (
        flux[:, :, :-1].spatial_downsample(2).sum(axis=None)[0]
        == flux[:, :, :-2].sum(axis=None)[0]
    )
    assert (flux[:, :, :-1].spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux[:, :, :-2].uncertainty.array ** 2).sum(axis=None).round()


def test_bool_cube():
    """Test BoolCube methods"""
    true_bool_array = np.ones(32).reshape((2, 4, 4)).astype(bool)
    assert "BoolCube (2, 4, 4)" in repr(BoolCube(true_bool_array))
    false_bool_array = ~true_bool_array
    mixed_bool_array_same = np.array([[np.ones(4), np.zeros(4)] * 2] * 2, dtype=bool)
    mixed_bool_array_opposite = mixed_bool_array_same.copy()
    mixed_bool_array_opposite[1] = ~mixed_bool_array_opposite[0]

    assert all(BoolCube(true_bool_array).downsample(2))
    assert all(~BoolCube(false_bool_array).downsample(2))
    assert (
        BoolCube(mixed_bool_array_same).downsample(2).array == mixed_bool_array_same[0]
    ).all()
    assert all(BoolCube(mixed_bool_array_opposite).downsample(2))

    false_append_true = np.append(false_bool_array, np.ones(16))
    false_append_true = false_append_true.reshape((3, 4, 4))
    assert all(BoolCube(false_append_true).downsample(3))


def test_bit_cube():
    """Test BitwiseCube methods"""
    from lkdata import BitwiseCube, BitwiseSeriesCollection, BitwiseSeries

    def strip(string):
        return string.replace(" ", "").replace("\n", "")

    flags = np.arange(32)
    flags = flags.reshape((2, 4, 4))
    code_dict = {i: f"C{i}" for i in [1, 2, 4, 8, 16]}
    bitcube = BitwiseCube(flags, codes=code_dict)
    assert (bitcube.array == flags).all()
    # Default repr shows data as given
    bitwise_str = strip(bitcube.styler.to_string())
    comp = "col0123row00123145672891011312131415"
    assert bitwise_str == "col0123row00123145672891011312131415"
    # bitset repr shows codes individually, without description
    bitcube.values_display = "bitset"
    bitset_str = strip(bitcube.styler.to_string())
    comp = (
        "col0123row0{}{1}{2}{1,2}1{4}{1,4}{2,4}{1,2,4}"
        "2{8}{1,8}{2,8}{1,2,8}3{4,8}{1,4,8}{2,4,8}{1,2,4,8}"
    )
    assert bitset_str == comp
    # Detailed repr parses and replaces codes with descriptions in given code_dict
    bitcube.values_display = "detailed"
    detailed_str = strip(bitcube.styler.to_string())
    comp = (
        "col0123row0{}{1:'C1'}{2:'C2'}{1:'C1',2:'C2'}1{4:'C4'}{1:'C1',4:'C4'}"
        "{2:'C2',4:'C4'}{1:'C1',2:'C2',4:'C4'}2{8:'C8'}{1:'C1',8:'C8'}"
        "{2:'C2',8:'C8'}{1:'C1',2:'C2',8:'C8'}3{4:'C4',8:'C8'}"
        "{1:'C1',4:'C4',8:'C8'}{2:'C2',4:'C4',8:'C8'}"
        "{1:'C1',2:'C2',4:'C4',8:'C8'}"
    )
    assert detailed_str == comp
    bitcube.codes = {i: f"New{i}" for i in [1, 2, 4, 8, 16]}
    detailed_str = strip(bitcube.styler.to_string())
    comp = (
        "col0123row0{}{1:'New1'}{2:'New2'}{1:'New1',2:'New2'}"
        "1{4:'New4'}{1:'New1',4:'New4'}{2:'New2',4:'New4'}"
        "{1:'New1',2:'New2',4:'New4'}2{8:'New8'}{1:'New1',8:'New8'}"
        "{2:'New2',8:'New8'}{1:'New1',2:'New2',8:'New8'}"
        "3{4:'New4',8:'New8'}{1:'New1',4:'New4',8:'New8'}{2:'New2',"
        "4:'New4',8:'New8'}{1:'New1',2:'New2',4:'New4',8:'New8'}"
    )
    assert detailed_str == comp
    # reset codes
    bitcube.codes = code_dict

    # Downsampling should combine codes without repetition (bitwise or)
    assert (bitcube.downsample(2).array == np.arange(16, 32).reshape(4, 4)).all()
    assert (
        bitcube.spatial_downsample(2).array
        == np.array([[[5, 7], [13, 15]], [[21, 23], [29, 31]]])
    ).all()

    # Frames
    assert isinstance(bitcube[:, [0, 1, 2, 3], :], BitwiseSeriesCollection)
    bitframe = bitcube[:, [0, 1, 2, 3], :]
    assert bitframe.shape == (2, 16)
    # Ensure codes and values_display transferred to derivative product
    detailed_str = strip(bitframe.styler.to_string())
    comp = (
        "series0123456789101112131415row0000111122223333"
        "col0123012301230123time_index0{}{1:'C1'}{2:'C2'}{1:'C1',2:'C2'}"
        "{4:'C4'}{1:'C1',4:'C4'}{2:'C2',4:'C4'}{1:'C1',2:'C2',4:'C4'}{8:'C8'}"
        "{1:'C1',8:'C8'}{2:'C2',8:'C8'}{1:'C1',2:'C2',8:'C8'}{4:'C4',8:'C8'}"
        "{1:'C1',4:'C4',8:'C8'}{2:'C2',4:'C4',8:'C8'}{1:'C1',2:'C2',4:'C4',"
        "8:'C8'}1{16:'C16'}{1:'C1',16:'C16'}{2:'C2',16:'C16'}{1:'C1',2:'C2',"
        "16:'C16'}{4:'C4',16:'C16'}{1:'C1',4:'C4',16:'C16'}{2:'C2',4:'C4',"
        "16:'C16'}{1:'C1',2:'C2',4:'C4',16:'C16'}{8:'C8',16:'C16'}{1:'C1',"
        "8:'C8',16:'C16'}{2:'C2',8:'C8',16:'C16'}{1:'C1',2:'C2',8:'C8',16:'C16'}"
        "{4:'C4',8:'C8',16:'C16'}{1:'C1',4:'C4',8:'C8',16:'C16'}{2:'C2',4:'C4',"
        "8:'C8',16:'C16'}{1:'C1',2:'C2',4:'C4',8:'C8',16:'C16'}"
    )
    assert detailed_str == comp

    bitframe = BitwiseSeriesCollection(np.arange(0, 32).reshape(2, 16))
    # New bitframe has no codes_dict, detailed display should be the same as bitset
    bitframe.values_display = "bitset"
    bitset_str = strip(bitframe.styler.to_string())
    comp = (
        "series(0,)(1,)(2,)(3,)(4,)(5,)(6,)(7,)(8,)(9,)(10,)(11,)(12,)(13,)"
        "(14,)(15,)time_index0{}{1}{2}{1,2}{4}{1,4}{2,4}{1,2,4}{8}"
        "{1,8}{2,8}{1,2,8}{4,8}{1,4,8}{2,4,8}{1,2,4,8}1{16}{1,16}{2,16}"
        "{1,2,16}{4,16}{1,4,16}{2,4,16}{1,2,4,16}{8,16}{1,8,16}{2,8,16}"
        "{1,2,8,16}{4,8,16}{1,4,8,16}{2,4,8,16}{1,2,4,8,16}"
    )
    assert bitset_str == comp
    bitframe.values_display = "detailed"
    detailed_str = strip(bitframe.styler.to_string())
    assert detailed_str == bitset_str
    # The values should match the derivative product from the cube
    assert (bitframe.array == bitcube[:, [0, 1, 2, 3], :]).all(axis=None)

    # Series
    assert isinstance(bitcube[:, 0, 0], BitwiseSeries)
    bitseries = BitwiseSeries(np.arange(0, 16), codes=code_dict)
    bitwise_str = strip(bitseries._repr_html_())
    comp = "00112233445566778899101011111212131314141515"
    assert comp in bitwise_str
    bitseries.values_display = "bitset"
    bitset_str = strip(bitseries._repr_html_())
    comp = (
        "0{}1{1}2{2}3{1,2}4{4}5{1,4}6{2,4}7{1,2,4}8{8}9{1,8}"
        "10{2,8}11{1,2,8}12{4,8}13{1,4,8}14{2,4,8}15{1,2,4,8}"
    )
    assert comp in bitset_str
    bitseries.values_display = "detailed"
    detailed_str = strip(bitseries._repr_html_())
    comp = (
        "0{}1{1:'C1'}2{2:'C2'}3{1:'C1',2:'C2'}4{4:'C4'}"
        "5{1:'C1',4:'C4'}6{2:'C2',4:'C4'}7{1:'C1',2:'C2',4:'C4'}"
        "8{8:'C8'}9{1:'C1',8:'C8'}10{2:'C2',8:'C8'}"
        "11{1:'C1',2:'C2',8:'C8'}12{4:'C4',8:'C8'}"
        "13{1:'C1',4:'C4',8:'C8'}14{2:'C2',4:'C4',8:'C8'}"
        "15{1:'C1',2:'C2',4:'C4',8:'C8'}"
    )
    assert comp in detailed_str
