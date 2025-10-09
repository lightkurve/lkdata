"""Test Uncetainty"""
import numpy as np
from lkdata.utils.uncertainty import Uncertainty


def test_init():
    data = np.ones((100, 10, 14))
    err = np.ones((100, 10, 14))
    uncertainty = Uncertainty(err)
    uncertainty.parent_nddata = data

    assert uncertainty.shape == (100, 10, 14)
    assert uncertainty.array.shape == (100, 10, 14)


def test_math():
    data = np.ones((100, 10, 14)) * 2
    err = np.ones((100, 10, 14))
    uncertainty = Uncertainty(err)
    uncertainty.parent_nddata = data

    add = uncertainty._propagate_add(uncertainty, data, 0)
    assert (add == np.sqrt(2)).all()

    sub = uncertainty._propagate_subtract(uncertainty, data, 0)
    assert (sub == np.sqrt(2)).all()

    mult = uncertainty._propagate_multiply(uncertainty, data * 2, 0)
    assert (mult == np.sqrt(8)).all()

    div = uncertainty._propagate_divide(uncertainty, data / 2, 0)
    assert (div == np.sqrt(8 / 2**4)).all()
