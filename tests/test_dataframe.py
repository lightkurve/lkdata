"""Unit tests for lkdata.DataFrame"""
import numpy as np
import pytest

from lkdata import (
    DataFrame,
    DataSeries,
    BoolFrame,
    BoolSeries,
    BitwiseFrame,
    BitwiseSeries,
)


def test_dataframe_init():
    """DataFrame initialization tests"""
    data = np.random.rand(10, 5)
    df = DataFrame(data)
    assert isinstance(df, DataFrame)
    assert df.shape == (10, 5)
    assert np.all(df.values == data)


def test_boolframe_init():
    """BoolFrame initialization tests"""
    data = np.random.choice([True, False], size=(10, 5))
    bf = BoolFrame(data)
    assert isinstance(bf, BoolFrame)
    assert bf.shape == (10, 5)
    assert np.all(bf.values == data)


def test_bitwiseframe_init():
    """BitwiseFrame initialization tests"""
    data = np.random.randint(0, 256, size=(10, 5))
    bwf = BitwiseFrame(data)
    assert isinstance(bwf, BitwiseFrame)
    assert bwf.shape == (10, 5)
    assert np.all(bwf.values == data)


def test_dataframe_getitem():
    """DataFrame getitem tests"""
    df = DataFrame(np.random.rand(10, 5))
    assert isinstance(df[0], DataFrame)
    assert isinstance(df[:5], DataFrame)
    assert isinstance(df[[0, 1]], DataFrame)
    assert isinstance(df[0, 1:3], DataFrame)
    assert isinstance(df[:, 0], DataSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = df[1, "two"]


def test_boolframe_getitem():
    bool_df = BoolFrame(np.random.choice([True, False], size=(10, 5)))
    assert isinstance(bool_df[0], BoolFrame)
    assert isinstance(bool_df[:5], BoolFrame)
    assert isinstance(bool_df[[0, 1]], BoolFrame)
    assert isinstance(bool_df[0, 1:3], BoolFrame)
    assert isinstance(bool_df[:, 0], BoolSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = bool_df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = bool_df[1, "two"]


def test_bitwiseframe_getitem():
    bitwise_df = BitwiseFrame(np.random.randint(0, 256, size=(10, 5)))
    assert isinstance(bitwise_df[0], BitwiseFrame)
    assert isinstance(bitwise_df[:5], BitwiseFrame)
    assert isinstance(bitwise_df[[0, 1]], BitwiseFrame)
    assert isinstance(bitwise_df[0, 1:3], BitwiseFrame)
    assert isinstance(bitwise_df[:, 0], BitwiseSeries)
    with pytest.raises(TypeError, match="non-integer key"):
        _ = bitwise_df["one"]
    with pytest.raises(ValueError, match="integer, integer"):
        _ = bitwise_df[1, "two"]
