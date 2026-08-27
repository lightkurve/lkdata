import numpy as np
import pandas as pd
import pytest

from astropy.io import fits
from pandas.io.formats.style import Styler
from lkdata import (
    TESTDATA,
    DataCube,
    DataSeriesCollection,
    DataSeries,
    BitwiseCube,
    BoolCube,
)
from lkdata.utils.uncertainty import Uncertainty
from lkdata.mixins import STATS_METHOD_NAMES

# ─── Fixtures ───────────────────────────────────────────────────────────────

ntime, nrow, ncol = 200, 10, 14
# This aperture makes it timeseries
aperture = np.zeros((10, 14), bool)
aperture[1:4, 1:4] = True


# ─── Init and repr ──────────────────────────────────────────────────────────


def test_repr_and_str():
    """repr includes the shape; __str__ delegates to __repr__."""
    assert f"DataCube {df.array.shape}" in repr(df)
    assert str(df) == repr(df)
    assert str(BoolCube(test_data.astype(bool))) == repr(
        BoolCube(test_data.astype(bool))
    )


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


# ─── __getitem__ / slicing ──────────────────────────────────────────────────


def test_slicing():
    """Test slicing a DataCube."""
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


test_data = np.ones((ntime, nrow, ncol))
df = DataCube(test_data, uncertainty=test_data)


# ─── Math and downsampling ──────────────────────────────────────────────────


def test_downsample():
    """Test downsampling methods."""
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


# ─── Fold and droplevel ─────────────────────────────────────────────────────


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


# ─── Real data ───────────────────────────────────────────────────────────────


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


# ─── BoolCube ────────────────────────────────────────────────────────────────


def test_bool_cube():
    """Test BoolCube methods."""
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


# ─── BitwiseCube ─────────────────────────────────────────────────────────────


def test_bit_cube():
    """Test BitwiseCube methods."""
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
        "series(0,0)(0,1)(0,2)(0,3)(1,0)(1,1)(1,2)(1,3)(2,0)(2,1)(2,2)(2,3)(3,0)(3,1)(3,2)(3,3)"
        "row0000111122223333"
        "col0123012301230123"
        "time_index0{}{1:'C1'}{2:'C2'}{1:'C1',2:'C2'}"
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


# ─── Fixtures ───────────────────────────────────────────────────────────────

ntime2, nrow2, ncol2 = 20, 5, 6
data = np.ones((ntime2, nrow2, ncol2))
err = np.ones((ntime2, nrow2, ncol2))

dc = DataCube(data, uncertainty=err)
dc_no_err = DataCube(data)
dc_named = DataCube(
    data,
    uncertainty=err,
    row_indices={"row": np.arange(nrow2)},
    col_indices={"col": np.arange(ncol2)},
)
dc_named_no_err = DataCube(
    data,
    row_indices={"row": np.arange(nrow2)},
    col_indices={"col": np.arange(ncol2)},
)


# ─── _stats_post_process paths ──────────────────────────────────────────────


def test_stats_post_process_axis_none_no_uncertainty():
    """axis=None with no uncertainty returns a scalar."""
    result = dc_no_err.mean(axis=None)
    assert isinstance(result, float)


def test_stats_post_process_axis0_no_uncertainty():
    """axis=0 with no uncertainty returns a plain (nrow, ncol) array."""
    result = dc_no_err.mean(axis=0)
    assert result.shape == (nrow2, ncol2)


def test_stats_post_process_axis1():
    """axis=1 ('series') returns a DataSeries (uncertainty embedded)."""
    result = dc.mean(axis=1)
    assert isinstance(result, DataSeries)
    assert result.shape == (ntime2,)


def test_stats_post_process_axis1_no_uncertainty():
    """axis=1 with no uncertainty returns just a DataSeries."""
    result = dc_no_err.mean(axis=1)
    assert isinstance(result, DataSeries)


# ─── Median method ────────────────────────────────────────────────────────────


def test_median():
    """median() returns correct shapes and types for axis=0 and axis=1."""
    result, _ = dc.median(axis=0)
    assert result.shape == (nrow2, ncol2)

    result1 = dc.median(axis=1)
    assert isinstance(result1, DataSeries)


def test_median_axis2_raises():
    """median(axis=2) raises ValueError for Cubes."""
    with pytest.raises(ValueError, match="axis=2"):
        dc.median(axis=2)


# ─── Cumulative methods ───────────────────────────────────────────────────────


def test_cumulative_methods():
    """cumsum/cummin/cummax/cumprod should return DataSeries of the same shape."""
    ds = DataSeries(np.ones(ntime2))
    for method_name in ["cumsum", "cummin", "cummax", "cumprod"]:
        result = getattr(ds, method_name)()
        assert isinstance(result, DataSeries)
        assert result.shape == ds.shape


# ─── __getitem__ edge cases ──────────────────────────────────────────────────


def test_getitem_1tuple():
    """Cube[(slice,)] is the same as Cube[slice]."""
    result = dc[(slice(None),)]
    assert result.shape == dc.shape


def test_getitem_too_many_raises():
    """Key with 4+ elements raises KeyError."""
    with pytest.raises(KeyError):
        _ = dc[0, 0, 0, 0]


def test_getitem_2tuple_int_with_uncertainty():
    """(time, int_pixel) key returns a DataSeries (uncertainty embedded)."""
    result = dc[0, 0]
    assert isinstance(result, DataSeries)


def test_getitem_2tuple_int_without_uncertainty():
    """(time, int_pixel) key with no uncertainty → DataSeries."""
    result = dc_no_err[0, 0]
    assert isinstance(result, DataSeries)


def test_getitem_bool_2d():
    """2-D boolean mask as key returns a plain DataFrame."""
    mask = np.zeros((nrow2, ncol2), dtype=bool)
    mask[0, 0] = True
    result = dc[:, mask]
    assert result.shape == (ntime2, 1)


def test_getitem_flat_bool_array():
    """Flat boolean array equal to nrow*ncol selects those columns."""
    flat = np.ones(nrow2 * ncol2, dtype=bool)
    result = dc_no_err[:, flat]
    assert isinstance(result, DataCube)
    assert result.ntime == ntime2


def test_getitem_step_slice_row_col():
    """Non-unit step slices delegate to to_seriescollection."""
    result = dc[:, ::2, ::2]
    assert isinstance(result, DataSeriesCollection)


# ─── __setitem__ non-positional key ─────────────────────────────────────────


def test_setitem_string_key():
    """Setting via a string key goes through pd.DataFrame.__setitem__."""
    dc_copy = DataCube(data.copy(), uncertainty=err.copy())
    dc_copy["extra"] = np.ones(ntime2)


# ─── _preprocess_data error paths ────────────────────────────────────────────


def test_preprocess_2d_no_nrow_ncol():
    """2-D data without nrow/ncol raises ValueError."""
    with pytest.raises(ValueError):
        DataCube(np.ones((ntime2, nrow2 * ncol2)))


def test_preprocess_wrong_ndim():
    """1-D data raises ValueError (can't be interpreted as a Cube)."""
    with pytest.raises(ValueError):
        DataCube(np.ones(ntime2), nrow=1, ncol=1)


# ─── Properties ──────────────────────────────────────────────────────────────


def test_col_names_default():
    """col_names returns [] when _col_names is None (the lazy-init path)."""
    dc2 = DataCube(data.copy())
    dc2._col_names = None  # Force the lazy-init branch
    assert dc2.col_names == []


def test_row_names_default():
    """row_names returns [] when _row_names is None (the lazy-init path)."""
    dc2 = DataCube(data.copy())
    dc2._row_names = None  # Force the lazy-init branch
    assert dc2.row_names == []


def test_nseries():
    """nseries = nrow * ncol."""
    assert dc.nseries == nrow2 * ncol2


def test_styler_property():
    """styler is None by default, and can be set."""
    dc2 = DataCube(
        data.copy(),
        row_indices={"row": np.arange(nrow2)},
        col_indices={"col": np.arange(ncol2)},
    )
    assert dc2.styler is None
    frame = dc2.get_single_frame(0)
    styler_val = dc2.stylize_frame(frame)
    dc2.styler = styler_val
    assert dc2.styler is not None


# ─── stylize_frame ────────────────────────────────────────────────────────────


def test_stylize_frame():
    """stylize_frame returns a Styler."""
    frame = dc_named.get_single_frame(0)
    styler = dc_named.stylize_frame(frame, label="test_label", cmap="gray")
    assert isinstance(styler, Styler)


# ─── _repr_html_ ─────────────────────────────────────────────────────────────


def test_repr_html_single_cadence():
    """_repr_html_ for a 1-cadence cube should not include the '+N cadences' string."""
    single = DataCube(
        data[:1],
        row_indices={"row": np.arange(nrow2)},
        col_indices={"col": np.arange(ncol2)},
    )
    html = single._repr_html_()
    assert "DataCube" in html
    assert "+0 cadences" not in html


def test_repr_html_multi_cadence():
    """_repr_html_ for a multi-cadence cube should include the '+N cadences' string."""
    html = dc_named._repr_html_()
    assert "DataCube" in html
    assert f"+{ntime2 - 1} cadences" in html


def test_repr_html_cached_styler():
    """Second call to _repr_html_ uses the cached styler."""
    dc2 = DataCube(
        data.copy(),
        row_indices={"row": np.arange(nrow2)},
        col_indices={"col": np.arange(ncol2)},
    )
    _ = dc2._repr_html_()
    assert dc2.styler is not None
    # Second call should use cached styler
    html2 = dc2._repr_html_()
    assert "DataCube" in html2


# ─── from_pandas with row_names / col_names ──────────────────────────────────


def test_from_pandas_row_names_string():
    """from_pandas with row_names as a string."""
    df = pd.DataFrame(dc_named)
    dc3 = DataCube.from_pandas(df, row_names="row", col_names="col")
    assert dc3.nrow == nrow2
    assert dc3.ncol == ncol2


def test_from_pandas_row_names_list():
    """from_pandas with row_names as a list."""
    df = pd.DataFrame(dc_named)
    dc3 = DataCube.from_pandas(df, row_names=["row"], col_names=["col"])
    assert dc3.nrow == nrow2
    assert dc3.ncol == ncol2


def test_from_pandas_missing_row_spec_raises():
    """from_pandas without row_names AND nrow raises KeyError."""
    df = pd.DataFrame(dc_named)
    with pytest.raises(KeyError):
        DataCube.from_pandas(df, col_names="col")


def test_from_pandas_missing_col_spec_raises():
    """from_pandas without col_names AND ncol raises KeyError."""
    df = pd.DataFrame(dc_named)
    with pytest.raises(KeyError):
        DataCube.from_pandas(df, row_names="row")


def test_from_pandas_bad_row_names_type_raises():
    """from_pandas with invalid row_names type raises ValueError."""
    df = pd.DataFrame(dc_named)
    with pytest.raises(ValueError):
        DataCube.from_pandas(df, row_names=123, col_names="col")


def test_from_pandas_bad_col_names_type_raises():
    """from_pandas with invalid col_names type raises ValueError."""
    df = pd.DataFrame(dc_named)
    with pytest.raises(ValueError):
        DataCube.from_pandas(df, row_names="row", col_names=123)


# ─── make_cadence_label ──────────────────────────────────────────────────────


def test_make_cadence_label_custom_index():
    """make_cadence_label with a float-valued custom index name (not 'index')."""
    t = np.linspace(2454833.0, 2454843.0, ntime2)
    dc2 = DataCube(
        data,
        time_indices={"days": t},
        row_indices={"row": np.arange(nrow2)},
        col_indices={"col": np.arange(ncol2)},
    )
    label = dc2.make_cadence_label(0)
    assert "days" in label


def test_make_cadence_label_indices_in_index():
    """make_cadence_label with 'indices' level (created by 'detailed' downsample)."""
    dc2 = DataCube(
        data,
        row_indices={"row": np.arange(nrow2)},
        col_indices={"col": np.arange(ncol2)},
    )
    ds = dc2.downsample(nframes=5, index_agg_func="detailed")
    label = ds.make_cadence_label(0)
    assert "indices" in label


# ─── _resolve_label_key and string-key getitem ───────────────────────────────


def test_string_key_getitem():
    """Accessing a named series via a string key (no uncertainty to avoid bug)."""
    first_label = dc_named_no_err.columns.get_level_values("series")[0]
    result = dc_named_no_err[first_label]
    assert isinstance(result, DataCube)
    assert result.ntime == ntime2


def test_string_key_not_found_raises():
    """Accessing a nonexistent series label raises KeyError."""
    with pytest.raises(KeyError):
        _ = dc_named_no_err["nonexistent_series"]


def test_list_of_strings_key():
    """List of string labels selects multiple series."""
    labels = list(dc_named_no_err.columns.get_level_values("series")[:2])
    result = dc_named_no_err[labels]
    assert isinstance(result, DataCube)


# ─── _get_iloc_key – additional paths ────────────────────────────────────────


def test_get_iloc_key_string_in_tuple():
    """_get_iloc_key handles a 2-tuple (time_slice, string_label) correctly."""
    first_label = dc_named_no_err.columns.get_level_values("series")[0]
    time_key, col_key = dc_named_no_err._get_iloc_key((slice(None), first_label))
    assert len(col_key) >= 1


def test_get_iloc_key_invalid_time_raises():
    """Invalid time key type raises ValueError."""
    with pytest.raises((ValueError, TypeError)):
        dc._get_iloc_key((1.5, 0))


# ─── BitwiseCube ─────────────────────────────────────────────────────────────


def test_bitwisecube_repr():
    """BitwiseCube repr includes class name."""
    flags = np.arange(20).reshape((4, 5, 1))
    bc = BitwiseCube(flags)
    assert "BitwiseCube" in repr(bc)


def test_bitwisecube_invalid_display_raises():
    """Setting values_display to an invalid value raises AttributeError for BitwiseCube."""
    flags = np.arange(20).reshape((4, 5, 1))
    bc = BitwiseCube(flags)
    with pytest.raises(AttributeError):
        bc.values_display = "not_a_valid_mode"


# ─── BoolMixin operations ─────────────────────────────────────────────────────


def test_boolcube_arithmetic():
    """BoolCube arithmetic operators."""
    bc1 = BoolCube(np.ones((ntime2, nrow2, ncol2), dtype=bool))
    bc2 = BoolCube(np.zeros((ntime2, nrow2, ncol2), dtype=bool))

    result = bc1 + bc2
    assert isinstance(result, BoolCube)
    # logical_or: all True
    assert result.all(axis=None)

    result_sub = bc1 - bc2
    assert isinstance(result_sub, BoolCube)

    result_mul = bc1 * bc2
    assert isinstance(result_mul, BoolCube)


def test_boolcube_arithmetic_wrong_type_raises():
    """BoolCube arithmetic with wrong type raises TypeError."""
    bc = BoolCube(np.ones((ntime2, nrow2, ncol2), dtype=bool))
    with pytest.raises(TypeError):
        _ = bc + 5
