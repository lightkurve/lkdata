"""Test Uncetainty"""
import numpy as np
from lkdata.utils.uncertainty import Uncertainty, from_variance_for_mean


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


def test_from_variance_for_mean_axis_none():
    """from_variance_for_mean with axis=None collapses over all elements."""
    x = np.ones((3, 4))
    result = from_variance_for_mean(x, axis=None)
    # sum(1) == 12, sqrt(12) / 12 == 1/sqrt(12)
    assert np.isclose(result, 1.0 / np.sqrt(12))


def test_uncertainty_propagate_min_axis_none():
    """Uncertainty propagation for min over all elements via _propagate_collapse."""
    data = np.arange(1, 13, dtype=float).reshape(3, 4)
    err = np.ones_like(data)
    u = Uncertainty(err)
    u.parent_nddata = data

    # Check that axis=None (collapse) returns a single float
    result = u._propagate_collapse(np.min, axis=None)
    assert np.isscalar(result) or result.ndim == 0


def test_uncertainty_propagate_add_sub_other_none():
    """Propagation when the other uncertainty array is None."""
    data = np.ones((4, 4))
    err = np.ones((4, 4))
    u_self = Uncertainty(err)
    u_self.parent_nddata = data
    u_other = Uncertainty(None)
    u_other.parent_nddata = data

    result = u_self._propagate_add(u_other, data, 0)
    # other has no array → treated as 0
    assert (result == 1.0).all()


def test_uncertainty_propagate_multiply_other_none():
    """Propagation for multiplication when other uncertainty is None."""
    data = np.ones((4, 4)) * 2
    err = np.ones((4, 4))
    u_self = Uncertainty(err)
    u_self.parent_nddata = data
    u_other = Uncertainty(None)
    u_other.parent_nddata = data

    result = u_self._propagate_multiply(u_other, data, 0)
    assert result is not None


def test_uncertainty_propagate_divide_other_none():
    """Propagation for division when other uncertainty is None."""
    data = np.ones((4, 4)) * 2
    err = np.ones((4, 4))
    u_self = Uncertainty(err)
    u_self.parent_nddata = data
    u_other = Uncertainty(None)
    u_other.parent_nddata = data

    result = u_self._propagate_divide(u_other, data, 0)
    assert result is not None


def test_uncertainty_propagate_with_nonzero_correlation():
    """Propagation with a non-zero correlation factor."""
    data = np.ones((4, 4)) * 2
    err = np.ones((4, 4))
    u = Uncertainty(err)
    u.parent_nddata = data
    u2 = Uncertainty(err.copy())
    u2.parent_nddata = data

    result = u._propagate_add(u2, data, correlation=0.5)
    assert result is not None
