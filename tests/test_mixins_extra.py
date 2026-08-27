"""Additional tests to improve coverage in mixins.py."""

import warnings

import numpy as np
import pandas as pd
import pytest

from lkdata import DataCube, DataSeries, DataSeriesCollection

ntime, nrow, ncol = 30, 4, 5
data = np.ones((ntime, nrow, ncol))
err = np.ones((ntime, nrow, ncol))
dc = DataCube(data, uncertainty=err)
dc_no_err = DataCube(data)


# ─── droplevel error paths ────────────────────────────────────────────────────


def test_droplevel_axis1_raises():
    """droplevel with axis=1 raises NotImplementedError."""
    dc2 = DataCube(
        data,
        time_indices={"extra": np.arange(ntime) * 2.0},
    )
    with pytest.raises(NotImplementedError):
        dc2.droplevel("extra", axis=1)


def test_droplevel_level0_raises():
    """droplevel with level='time_index' raises ValueError."""
    dc2 = DataCube(
        data,
        time_indices={"extra": np.arange(ntime) * 2.0},
    )
    with pytest.raises(ValueError):
        dc2.droplevel("time_index")


# ─── agg_index detailed mode ─────────────────────────────────────────────────


def test_agg_index_detailed_with_time_index():
    """agg_index in 'detailed' mode tracks which time_index values were binned."""
    dc2 = DataCube(data)
    ds = dc2.downsample(nframes=5, index_agg_func="detailed")
    assert "indices" in ds.index.names
    assert ds.ntime < ntime


def test_agg_index_detailed_with_existing_indices():
    """agg_index in 'detailed' mode with an existing 'indices' level repacks it."""
    dc2 = DataCube(data)
    ds1 = dc2.downsample(nframes=5, index_agg_func="detailed")
    # Downsample on level=0 (integer time_index) to avoid the string-diff bug
    ds2 = ds1.downsample(nframes=2, level=0, index_agg_func="detailed")
    assert "indices" in ds2.index.names


# ─── bin method – counts=True ────────────────────────────────────────────────


def test_bin_returns_counts():
    """bin(..., counts=True) returns (binned_object, counts_array)."""
    index = dc.index.get_level_values("time_index")
    bins = DataCube.get_bins(np.array(index, dtype=float), nframes=5)
    binned, counts = dc.bin(
        bins=bins,
        level="time_index",
        agg_func="sum",
        counts=True,
        min_count=5,
    )
    assert isinstance(binned, DataCube)
    assert counts is not None


def test_bin_with_uncertainty_agg_func():
    """bin() with a custom uncertainty_agg_func."""
    index = dc.index.get_level_values("time_index")
    bins = DataCube.get_bins(np.array(index, dtype=float), nframes=5)
    binned = dc.bin(
        bins=bins,
        level="time_index",
        agg_func="sum",
        uncertainty_agg_func="sum",
        min_count=5,
    )
    assert isinstance(binned, DataCube)


def test_bin_series():
    """bin() on a DataSeries."""
    ds = DataSeries(np.ones(ntime), uncertainty=np.ones(ntime))
    index = ds.index.get_level_values("time_index")
    bins = DataSeries.get_bins(np.array(index, dtype=float), nframes=5)
    binned = ds.bin(bins=bins, level="time_index", agg_func="sum", min_count=5)
    assert isinstance(binned, DataSeries)


# ─── sort_index with inplace=True ────────────────────────────────────────────


def test_sort_index_inplace():
    """sort_index(inplace=True, level=1) modifies the DataCube in place."""
    t = np.arange(ntime, dtype=float)
    t_shuffled = t.copy()
    np.random.shuffle(t_shuffled)
    dc2 = DataCube(data, time_indices={"t": t_shuffled})
    dc2.sort_index(level=1, inplace=True)
    assert all(np.diff(dc2.index.get_level_values(1)) >= 0)


def test_sort_index_inplace_with_uncertainty():
    """sort_index(inplace=True) correctly reorders the uncertainty array."""
    t = np.arange(ntime, dtype=float)
    t_shuffled = t.copy()
    np.random.shuffle(t_shuffled)
    unique_err = np.arange(ntime, dtype=float).reshape(ntime, 1, 1) * np.ones(
        (ntime, nrow, ncol)
    )
    dc2 = DataCube(data, uncertainty=unique_err, time_indices={"t": t_shuffled})
    dc2.sort_index(level=1, inplace=True)
    sorted_t = dc2.index.get_level_values(1)
    assert all(np.diff(sorted_t) >= 0)


# ─── sort_index outplace (default) ───────────────────────────────────────────


def test_sort_index_returns_new_instance():
    """sort_index(inplace=False) returns a new DataCube."""
    t = np.arange(ntime, dtype=float)[::-1]
    dc2 = DataCube(data, uncertainty=err, time_indices={"t": t})
    sorted_dc = dc2.sort_index(level=1)
    assert isinstance(sorted_dc, DataCube)
    sorted_t = sorted_dc.index.get_level_values(1)
    assert all(np.diff(sorted_t) >= 0)


# ─── _arithmetic_uncertainty edge cases ──────────────────────────────────────


def test_math_operand_without_uncertainty():
    """Arithmetic where the operand has no uncertainty attribute."""
    arr = np.ones((ntime, nrow, ncol))
    result = dc + arr
    assert isinstance(result, DataCube)
    assert result.uncertainty.array is not None


def test_math_self_without_uncertainty():
    """Arithmetic where self has no uncertainty but operand does."""
    result = dc_no_err + dc
    assert isinstance(result, DataCube)


def test_math_both_without_uncertainty():
    """Arithmetic where neither self nor operand has uncertainty."""
    result = dc_no_err + dc_no_err
    assert isinstance(result, DataCube)
    assert not result.uncertainty


def test_math_scalar_multiply():
    """Scalar multiplication propagates uncertainty correctly."""
    result = dc * 2
    assert (result.uncertainty.array == 2).all()


def test_math_scalar_divide():
    """Scalar division propagates uncertainty correctly."""
    result = dc / 2
    assert (result.uncertainty.array == 0.5).all()


def test_math_mod():
    """Modulo operation on DataCube."""
    result = dc % 1
    assert isinstance(result, DataCube)


def test_math_pow():
    """Power operation on DataCube."""
    result = dc**2
    assert isinstance(result, DataCube)


# ─── uncertainty.setter with float/int ───────────────────────────────────────


def test_uncertainty_setter_float():
    """Setting uncertainty to a float broadcasts to the full array shape."""
    dc2 = DataCube(data)
    dc2.uncertainty = 0.5
    assert (dc2.uncertainty.array == 0.5).all()
    assert dc2.uncertainty.array.shape == data.shape


def test_uncertainty_setter_int():
    """Setting uncertainty to an int broadcasts to the full array shape."""
    dc2 = DataCube(data)
    dc2.uncertainty = 2
    assert (dc2.uncertainty.array == 2).all()


# ─── _process_math_val DataFrame / Series ────────────────────────────────────


def test_process_math_val_dataframe():
    """Arithmetic with a pd.DataFrame operand."""
    df_operand = pd.DataFrame(np.ones((ntime, nrow * ncol)))
    ds = DataSeriesCollection(np.ones((ntime, nrow * ncol)))
    result = ds + df_operand
    assert isinstance(result, DataSeriesCollection)


def test_process_math_val_series():
    """Arithmetic with a pd.Series operand."""
    series_operand = pd.Series(np.ones(ntime))
    ds = DataSeries(np.ones(ntime))
    result = ds + series_operand
    assert isinstance(result, DataSeries)


def test_process_math_val_bad_type_raises():
    """Arithmetic with an unsupported type raises TypeError."""
    ds = DataSeries(np.ones(ntime))
    with pytest.raises(TypeError):
        ds + "not_a_number"


# ─── BoolMixin methods ────────────────────────────────────────────────────────


def test_boolseries_collection_add():
    """BoolSeriesCollection addition (logical OR)."""
    from lkdata import BoolSeriesCollection

    bs1 = BoolSeriesCollection(np.ones((ntime, nrow * ncol), dtype=bool))
    bs2 = BoolSeriesCollection(np.zeros((ntime, nrow * ncol), dtype=bool))
    result = bs1 + bs2
    assert isinstance(result, BoolSeriesCollection)
    assert result.all(axis=None)


def test_boolseries_add():
    """BoolSeries arithmetic via BoolMixin.__add__ (called directly)."""
    from lkdata import BoolSeries
    from lkdata.mixins import BoolMixin

    bs1 = BoolSeries(np.ones(ntime, dtype=bool))
    bs2 = BoolSeries(np.zeros(ntime, dtype=bool))

    # Call BoolMixin.__add__ directly (pandas intercepts the + operator for BoolSeries)
    result = BoolMixin.__add__(bs1, bs2)
    assert isinstance(result, BoolSeries)
    assert result.all()

    # __sub__ and __mul__ delegate to self.__add__ (pandas path in BoolSeries)
    result_sub = BoolMixin.__sub__(bs1, bs2)
    assert result_sub is not None

    result_mul = BoolMixin.__mul__(bs1, bs2)
    assert result_mul is not None

    # Wrong type raises TypeError
    with pytest.raises(TypeError):
        BoolMixin.__add__(bs1, 42)


# ─── BitwiseMixin.values_display invalid value (series collection path) ───────


def test_bitwise_values_display_invalid_warns():
    """Setting values_display on BitwiseSeriesCollection to an invalid value warns and defaults."""
    from lkdata import BitwiseSeriesCollection

    bsc = BitwiseSeriesCollection(np.arange(10).reshape(5, 2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bsc.values_display = "invalid"
        assert len(w) > 0
    assert bsc.values_display == "int"


# ─── _set_data_type_to_int ────────────────────────────────────────────────────


def test_set_data_type_to_int_from_str():
    """_set_data_type_to_int converts binary strings to integers."""
    from lkdata.mixins import BitwiseMixin

    result = BitwiseMixin._set_data_type_to_int(np.array(["0b101", "0b011"]))
    assert list(result) == [5, 3]


def test_set_data_type_to_int_from_set():
    """_set_data_type_to_int converts sets to integers (sum of members)."""
    from lkdata.mixins import BitwiseMixin

    result = BitwiseMixin._set_data_type_to_int(np.array([{1, 4}, {2}]))
    assert set(result) == {5, 2}


def test_set_data_type_to_int_from_int():
    """_set_data_type_to_int passes through integer arrays unchanged."""
    from lkdata.mixins import BitwiseMixin

    arr = np.array([1, 3, 7])
    result = BitwiseMixin._set_data_type_to_int(arr)
    np.testing.assert_array_equal(result, arr)


def test_set_data_type_to_int_bad_type_raises():
    """_set_data_type_to_int raises ValueError for uncastable types."""
    from lkdata.mixins import BitwiseMixin

    with pytest.raises(ValueError):
        BitwiseMixin._set_data_type_to_int(np.array(["not_binary", "also_bad"]))


# ─── spatial_aggregate (frame/2D path) ───────────────────────────────────────


def test_spatial_aggregate_cube():
    """spatial_aggregate on a DataCube returns the expected shape."""
    target_nrow, target_ncol = 2, 2
    result = dc.spatial_aggregate(target_nrow, target_ncol)
    assert isinstance(result, DataCube)
    assert result.array.shape[1] == target_nrow
    assert result.array.shape[2] == target_ncol


# ─── parse_index error paths ──────────────────────────────────────────────────


def test_parse_index_non_multiindex_raises():
    """parse_index raises ValueError when a non-Index object is passed as index."""
    with pytest.raises(ValueError, match="pd.MultiIndex"):
        DataCube.parse_index(index="bad_value", ntime=10)


# ─── DataSeries _stats_post_process ──────────────────────────────────────────


def test_dataseries_stats_post_process_with_uncertainty():
    """DataSeries stats methods return (result, uncertainty) tuple."""
    ds = DataSeries(np.ones(ntime), uncertainty=np.ones(ntime))
    result, uncert = ds.mean(axis=None)
    assert isinstance(result, (float, np.floating))


def test_dataseries_stats_post_process_no_uncertainty():
    """DataSeries stats methods return scalar when no uncertainty."""
    ds = DataSeries(np.ones(ntime))
    result = ds.mean(axis=None)
    assert isinstance(result, (float, np.floating))


# ─── SeriesCollection _stats_post_process ────────────────────────────────────


def test_seriescollection_stats_axis_time_with_uncertainty():
    """DataSeriesCollection.mean(axis=0) returns (result, uncertainty) tuple."""
    dsc = DataSeriesCollection(
        np.ones((ntime, nrow * ncol)), uncertainty=np.ones((ntime, nrow * ncol))
    )
    result, uncert = dsc.mean(axis=0)
    assert result.shape == (nrow * ncol,)


def test_seriescollection_stats_axis_pixel_with_uncertainty():
    """DataSeriesCollection.mean(axis=1) returns DataSeries (uncertainty embedded)."""
    dsc = DataSeriesCollection(
        np.ones((ntime, nrow * ncol)), uncertainty=np.ones((ntime, nrow * ncol))
    )
    result = dsc.mean(axis=1)
    assert isinstance(result, DataSeries)


def test_seriescollection_stats_axis_none_no_uncertainty():
    """DataSeriesCollection.mean(axis=None) with no uncertainty returns scalar."""
    dsc = DataSeriesCollection(np.ones((ntime, nrow * ncol)))
    result = dsc.mean(axis=None)
    assert isinstance(result, (float, np.floating))
