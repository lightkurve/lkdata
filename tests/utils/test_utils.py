import pytest

from lkdata.utils import (
    LightkurveWarning,
    LightkurveError,
)


def test_lightkurve_warning():
    with pytest.raises(LightkurveWarning):
        raise LightkurveWarning("Test warning")


def test_lightkurve_error():
    with pytest.raises(LightkurveError):
        raise LightkurveError("Test error")
