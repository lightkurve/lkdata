"""Unit tests for lkdata.DataFrame"""
import numpy as np
import pytest

from lkdata import (
    DataSeriesCollection,
    DataSeries,
    BoolSeriesCollection,
    BoolSeries,
    BitwiseSeriesCollection,
    BitwiseSeries,
)


def test_dataframe_init():
    """DataFrame initialization tests"""
    data = np.random.rand(10, 5)
    df = DataSeriesCollection(data)
    assert isinstance(df, DataSeriesCollection)
    assert df.shape == (10, 5)
    assert np.all(df.values == data)

    df = DataSeriesCollection(data, index=range(10, 20))
    assert "given_index" in df.index.names


def test_boolframe_init():
    """BoolFrame initialization tests"""
    data = np.random.choice([True, False], size=(10, 5))
    bf = BoolSeriesCollection(data)
    assert isinstance(bf, BoolSeriesCollection)
    assert bf.shape == (10, 5)
    assert np.all(bf.values == data)


def test_bitwiseframe_init():
    """BitwiseFrame initialization tests"""
    data = np.random.randint(0, 256, size=(10, 5))
    bwf = BitwiseSeriesCollection(data)
    assert isinstance(bwf, BitwiseSeriesCollection)
    assert bwf.shape == (10, 5)
    assert np.all(bwf.values == data)


def test_dataframe_getitem():
    """DataFrame getitem tests"""
    df = DataSeriesCollection(np.random.rand(10, 5))
    assert isinstance(df[0], DataSeriesCollection)
    assert isinstance(df[:5], DataSeriesCollection)
    assert isinstance(df[[0, 1]], DataSeriesCollection)
    assert isinstance(df[0, 1:3], DataSeriesCollection)
    assert isinstance(df[:, 0], DataSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = df[1, "two"]


def test_boolframe_getitem():
    bool_df = BoolSeriesCollection(np.random.choice([True, False], size=(10, 5)))
    assert isinstance(bool_df[0], BoolSeriesCollection)
    assert isinstance(bool_df[:5], BoolSeriesCollection)
    assert isinstance(bool_df[[0, 1]], BoolSeriesCollection)
    assert isinstance(bool_df[0, 1:3], BoolSeriesCollection)
    assert isinstance(bool_df[:, 0], BoolSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = bool_df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = bool_df[1, "two"]


def test_bitwiseframe_getitem():
    bitwise_df = BitwiseSeriesCollection(np.random.randint(0, 256, size=(10, 5)))
    assert isinstance(bitwise_df[0], BitwiseSeriesCollection)
    assert isinstance(bitwise_df[:5], BitwiseSeriesCollection)
    assert isinstance(bitwise_df[[0, 1]], BitwiseSeriesCollection)
    assert isinstance(bitwise_df[0, 1:3], BitwiseSeriesCollection)
    assert isinstance(bitwise_df[:, 0], BitwiseSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = bitwise_df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = bitwise_df[1, "two"]
