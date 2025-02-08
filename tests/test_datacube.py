import numpy as np
import pandas as pd
from astropy.io import fits
from lkdata import (
    TESTDATA,
    DataCube,
    DataFrame,
    DataSeries,
    BoolCube,
)
from lkdata.uncertainty import Uncertainty
from lkdata.mixins import STATS_METHOD_NAMES


def test_setup():
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

    assert df.array.shape == (ntime, nrow, ncol)
    assert np.allclose(df.array, test_data)

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

    assert isinstance(df[:, [0, 1], :2], DataFrame)
    assert df[:, [0, 1], :2].ntime == ntime
    assert df[:, [0, 1], :2].nseries == 4

    assert all(df[:, [0, 1], 0] == df[:, [0, 1], [0, 0]])
    assert isinstance(df[:, [0, 1], 0], DataFrame)
    assert df[:, [0, 1], 0].ntime == ntime
    assert df[:, [0, 1], 0].nseries == 2

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

    # DataFrame
    assert isinstance(df[:, :, 0], DataFrame)
    assert df[:, :, 0].ntime == ntime
    assert df[:, :, 0].shape == (ntime, nrow)

    # DataSeries
    assert isinstance(df[:, 1, 0], DataSeries)
    assert df[:, 1, 0].ntime == ntime
    assert df[:, 1, 0].shape == (ntime,)

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


def test_downsample():
    # Example usage
    ntime, nrow, ncol = 200, 10, 14
    test_data = np.ones((ntime, nrow, ncol))
    test_data = np.ones((ntime, nrow, ncol))
    df = DataCube(test_data, uncertainty=test_data)
    # df_err = ErrorCube(test_data)
    # Time downsample
    assert (df.downsample(2).array == 2).all()
    assert (df.downsample(4).uncertainty.array == 2).all()

    # Spatial downsample
    assert df.spatial_downsample(2).array.shape == (200, 5, 7)
    assert df.spatial_downsample((2, 1)).array.shape == (200, 5, 14)
    assert df.spatial_downsample((1, 2)).array.shape == (200, 10, 7)

    assert (df.spatial_downsample(2).array == 4).all()
    assert all(df.spatial_downsample((2, 1)) == 2)
    assert all(df.spatial_downsample((1, 2)) == 2)
    assert df[:, :, :-1].spatial_downsample(2).array.shape == (200, 5, 6)
    assert (
        df[:, :, :-1].spatial_downsample(2).array
        == df[:, :, :-2].spatial_downsample(2).array
    ).all()
    assert df[:, :-1, :].spatial_downsample(2).array.shape == (200, 4, 7)

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
        uncertainty=flux_err_array,
        time_indices={"btjd": time, "spacecraft_time": time - time_corr},
        row_indices={"row": row},
        col_indices={"column": col},
    )

    return flux, aper, bkg_aper, time_mask


def test_real_data():
    flux, aper, _, _ = make_test_data()

    assert isinstance(flux, DataCube)
    assert isinstance(flux.uncertainty, Uncertainty)

    assert flux.array.shape == (50, 6, 6)
    assert flux.uncertainty.array.shape == (50, 6, 6)

    assert flux.downsample(5).array.shape == (8, 6, 6)
    assert flux.downsample(5).uncertainty.array.shape == (8, 6, 6)

    assert isinstance(flux[:, aper], DataFrame)
    assert isinstance(flux[:, aper].uncertainty, Uncertainty)

    assert isinstance(flux[:, aper].sum(axis=1), DataSeries)
    assert isinstance(flux[:, aper].sum(axis=1).uncertainty, Uncertainty)

    assert flux.spatial_downsample(2).array.shape == (50, 3, 3)
    assert flux.spatial_downsample(2).uncertainty.shape == (50, 3, 3)

    assert flux.spatial_downsample(2).array[0, 0, 0] == flux[0, :2, :2].sum(axis=None)
    assert (
        flux.spatial_downsample(2).uncertainty.array[0, 0, 0]
        == ((flux[0, :2, :2].uncertainty.array ** 2).sum().sum()) ** 0.5
    )

    assert flux.spatial_downsample(2).array[0, -1, -1] == flux[0, -2:, -2:].sum(
        axis=None
    )
    assert (
        flux.spatial_downsample(2).uncertainty.array[0, -1, -1]
        == ((flux[0, -2:, -2:].uncertainty.array ** 2).sum(axis=None)) ** 0.5
    )

    assert flux.spatial_downsample(2).sum(axis=None) == flux.sum(axis=None)
    assert (flux.spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux.uncertainty.array**2).sum(axis=None).round()

    assert flux[:, :-1].spatial_downsample(2).sum(axis=None) == flux[:, :-2].sum(
        axis=None
    )
    assert (flux[:, :-1].spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux[:, :-2].uncertainty.array ** 2).sum(axis=None).round()
    assert flux[:, :, :-1].spatial_downsample(2).sum(axis=None) == flux[:, :, :-2].sum(
        axis=None
    )
    assert (flux[:, :, :-1].spatial_downsample(2).uncertainty.array ** 2).sum(
        axis=None
    ).round() == (flux[:, :, :-2].uncertainty.array ** 2).sum(axis=None).round()


def test_bool_cube():
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
    from lkdata import BitwiseCube, BitwiseFrame, BitwiseSeries

    def strip(string):
        return string.replace(" ", "").replace("\n", "")

    flags = np.arange(32)
    flags = flags.reshape((2, 4, 4))
    code_dict = {i: f"C{i}" for i in [1, 2, 4, 8, 16]}
    bitcube = BitwiseCube(flags, codes=code_dict)
    assert (bitcube.array == flags).all()
    # Default repr shows data as given
    bitwise_str = strip(bitcube.styler.to_string())
    assert bitwise_str == "col0123row00123145672891011312131415"
    # Parsed repr shows codes individually, without description
    bitcube.values_display = "parsed"
    parsed_str = strip(bitcube.styler.to_string())
    assert (
        parsed_str
        == "col0123row0{}{1}{2}{1,2}1{4}{1,4}{2,4}{1,2,4}2{8}{1,8}{2,8}{1,2,8}3{4,8}{1,4,8}{2,4,8}{1,2,4,8}"
    )
    # Detailed repr parses and replaces codes with descriptions in given code_dict
    bitcube.values_display = "detailed"
    detailed_str = strip(bitcube.styler.to_string())
    assert (
        detailed_str
        == "col0123row0{}{1:'C1'}{2:'C2'}{1:'C1',2:'C2'}1{4:'C4'}{1:'C1',4:'C4'}{2:'C2',4:'C4'}{1:'C1',2:'C2',4:'C4'}2{8:'C8'}{1:'C1',8:'C8'}{2:'C2',8:'C8'}{1:'C1',2:'C2',8:'C8'}3{4:'C4',8:'C8'}{1:'C1',4:'C4',8:'C8'}{2:'C2',4:'C4',8:'C8'}{1:'C1',2:'C2',4:'C4',8:'C8'}"
    )
    bitcube.codes = {i: f"New{i}" for i in [1, 2, 4, 8, 16]}
    detailed_str = strip(bitcube.styler.to_string())
    assert (
        detailed_str
        == "col0123row0{}{1:'New1'}{2:'New2'}{1:'New1',2:'New2'}1{4:'New4'}{1:'New1',4:'New4'}{2:'New2',4:'New4'}{1:'New1',2:'New2',4:'New4'}2{8:'New8'}{1:'New1',8:'New8'}{2:'New2',8:'New8'}{1:'New1',2:'New2',8:'New8'}3{4:'New4',8:'New8'}{1:'New1',4:'New4',8:'New8'}{2:'New2',4:'New4',8:'New8'}{1:'New1',2:'New2',4:'New4',8:'New8'}"
    )
    # reset codes
    bitcube.codes = code_dict

    # Downsampling should combine codes without repetition (bitwise or)
    assert (bitcube.downsample(2).array == np.arange(16, 32).reshape(4, 4)).all()
    assert (
        bitcube.spatial_downsample(2).array
        == np.array([[[5, 7], [13, 15]], [[21, 23], [29, 31]]])
    ).all()

    # Frames
    assert isinstance(bitcube[:, [0, 1, 2, 3], :], BitwiseFrame)
    bitframe = bitcube[:, [0, 1, 2, 3], :]
    assert bitframe.shape == (2, 16)
    # Ensure codes and values_display transferred to derivative product
    detailed_str = strip(bitframe.styler.to_string())
    assert (
        detailed_str
        == "series0123456789101112131415row0000111122223333col0123012301230123time_index0{}{1:'C1'}{2:'C2'}{1:'C1',2:'C2'}{4:'C4'}{1:'C1',4:'C4'}{2:'C2',4:'C4'}{1:'C1',2:'C2',4:'C4'}{8:'C8'}{1:'C1',8:'C8'}{2:'C2',8:'C8'}{1:'C1',2:'C2',8:'C8'}{4:'C4',8:'C8'}{1:'C1',4:'C4',8:'C8'}{2:'C2',4:'C4',8:'C8'}{1:'C1',2:'C2',4:'C4',8:'C8'}1{16:'C16'}{1:'C1',16:'C16'}{2:'C2',16:'C16'}{1:'C1',2:'C2',16:'C16'}{4:'C4',16:'C16'}{1:'C1',4:'C4',16:'C16'}{2:'C2',4:'C4',16:'C16'}{1:'C1',2:'C2',4:'C4',16:'C16'}{8:'C8',16:'C16'}{1:'C1',8:'C8',16:'C16'}{2:'C2',8:'C8',16:'C16'}{1:'C1',2:'C2',8:'C8',16:'C16'}{4:'C4',8:'C8',16:'C16'}{1:'C1',4:'C4',8:'C8',16:'C16'}{2:'C2',4:'C4',8:'C8',16:'C16'}{1:'C1',2:'C2',4:'C4',8:'C8',16:'C16'}"
    )

    bitframe = BitwiseFrame(np.arange(0, 32).reshape(2, 16))
    # New bitframe has no codes_dict, detailed display should be the same as parsed
    bitframe.values_display = "parsed"
    parsed_str = strip(bitframe.styler.to_string())
    assert (
        parsed_str
        == "01234567891011121314150{}{1}{2}{1,2}{4}{1,4}{2,4}{1,2,4}{8}{1,8}{2,8}{1,2,8}{4,8}{1,4,8}{2,4,8}{1,2,4,8}1{16}{1,16}{2,16}{1,2,16}{4,16}{1,4,16}{2,4,16}{1,2,4,16}{8,16}{1,8,16}{2,8,16}{1,2,8,16}{4,8,16}{1,4,8,16}{2,4,8,16}{1,2,4,8,16}"
    )
    bitframe.values_display = "detailed"
    detailed_str = strip(bitframe.styler.to_string())
    assert detailed_str == parsed_str
    # The values should match the derivative product from the cube
    assert (bitframe.array == bitcube[:, [0, 1, 2, 3], :]).all(axis=None)

    # Series
    assert isinstance(bitcube[:, 0, 0], BitwiseSeries)
    bitseries = BitwiseSeries(np.arange(0, 16), codes=code_dict)
    bitwise_str = strip(repr(bitseries))[29:-12]
    assert bitwise_str == "0011223344556677889910101111121213131414151"
    bitseries.values_display = "parsed"
    parsed_str = strip(repr(bitseries))[29:-12]
    assert (
        parsed_str
        == "0{}1{1}2{2}3{1,2}4{4}5{1,4}6{2,4}7{1,2,4}8{8}9{1,8}10{2,8}11{1,2,8}12{4,8}13{1,4,8}14{2,4,8}15{1,2,4,8}"
    )
    bitseries.values_display = "detailed"
    detailed_str = strip(repr(bitseries))[29:-12]
    assert (
        detailed_str
        == "0{}1{1:'C1'}2{2:'C2'}3{1:'C1',2:'C2'}4{4:'C4'}5{1:'C1',4:'C4'}6{2:'C2',4:'C4'}7{1:'C1',2:'C2',4:'C4'}8{8:'C8'}9{1:'C1',8:'C8'}10{2:'C2',8:'C8'}11{1:'C1',2:'C2',8:'C8'}12{4:'C4',8:'C8'}13{1:'C1',4:'C4',8:'C8'}14{2:'C2',4:'C4',8:'C8'}15{1:'C1',2:'C2',4:'C4',8:'C8'}"
    )
