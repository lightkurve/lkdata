import numpy as np
import pandas as pd
import pytest
from lkdata import DataSeries


@pytest.fixture
def ntime():
    return 100


@pytest.fixture
def time(ntime):
    return np.linspace(0, 100, ntime)


@pytest.fixture
def data(ntime):
    return np.random.normal(size=(ntime))


@pytest.fixture
def ds(data, time):
    return DataSeries(data, time_indices={"time": time})


def test_init(data, time, ds):
    """Test general initialization is as expected"""
    assert all(ds.time == time)
    assert all(ds.values == data)
    assert "time_index" in ds.index.names

    ds = DataSeries(data, index=time)
    assert "given_index" in ds.index.names


def test_init_w_dict(data, time, ds):
    data_dict = dict(zip(time, data))
    ds_comp = DataSeries(data_dict)

    assert all(ds_comp == ds)
    assert "key_index" in ds_comp.index.names
    assert all(ds_comp.key_index == time)


def test_init_w_name(data, time, ds):
    data_dict = {"flux1": data}
    ds_comp = DataSeries(data_dict, time_indices={"time": time})
    assert all(ds_comp == ds)


def test_init_w_dict_altindex(data, time, ds):
    index = pd.Index(range(100, 200), name="alt_index")
    data_dict = dict(zip(time, data))
    ds_comp = DataSeries(data_dict, index=index)
    assert all(ds_comp.values == ds.values)
    assert "key_index" in ds_comp.index.names
    assert "alt_index" in ds_comp.index.names


def test_init_w_dict_and_mismatchindex(data, time, ds):
    data_dict = dict(zip(time, data))
    index = pd.Index(time[::2])
    ds_comp = DataSeries(data_dict, index=index)

    assert all(ds_comp.values == ds[::2].values)


def test_getitem_int_returns_scalar(data, time, ds):
    """__getitem__ with an int key returns the value at that position."""
    result = ds[0]
    # For a MultiIndex DataSeries, integer key returns a sub-Series (label lookup)
    assert isinstance(result, (float, int, np.floating, np.integer, pd.Series))


def test_getitem_int_with_uncertainty_returns_tuple(data, time):
    """__getitem__ with an int key and uncertainty returns (value, uncertainty) tuple."""
    ds_err = DataSeries(data, uncertainty=data, time_indices={"time": time})
    result = ds_err[0]
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_repr_html(data, time, ds):
    """_repr_html_ should return None (it uses print) without raising."""
    # The method prints to stdout and returns None
    result = ds._repr_html_()
    assert result is None


def test_stats_post_process_uncertainty(data, time):
    """Stats methods return (result, uncertainty) tuple when uncertainty exists."""
    ds_err = DataSeries(data, uncertainty=data, time_indices={"time": time})
    result = ds_err.mean(axis=None)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_stats_post_process_no_uncertainty(data, time, ds):
    """Stats methods return a scalar when no uncertainty."""
    result = ds.mean(axis=None)
    assert isinstance(result, (float, int, np.floating))
