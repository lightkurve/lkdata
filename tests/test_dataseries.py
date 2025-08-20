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
