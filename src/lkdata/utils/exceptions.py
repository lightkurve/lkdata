"""Contains all lightkurve exceptions."""

__all__ = [
    "LightkurveError",
    "LightkurveWarning",
    "LightkurveDeprecationWarning",
]


class LightkurveError(Exception):
    """Class for Lightkurve exceptions."""

    pass


class LightkurveWarning(Warning):
    """Class for all Lightkurve warnings."""

    pass


class LightkurveDeprecationWarning(LightkurveWarning):
    """Class for all Lightkurve deprecation warnings."""

    pass
