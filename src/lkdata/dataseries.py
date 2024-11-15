"""Classes and tools for working with 1 dimensional data."""

import logging
from abc import ABC

import pandas as pd

from .mixins import (
    AggMixin,
    ConvenienceMixins,
    ErrorStatsMixin,
    MathMixin,
    StatsMixin,
    BoolStatsMixin,
    BitwiseMixin,
)

log = logging.getLogger()


class Series(
    ABC,
    MathMixin,
    AggMixin,
    ConvenienceMixins,
    pd.Series,
):
    """Abstract dataclass for series-like data with time and data"""

    _pd_class = pd.Series
    _user_kwargs = None

    def __init__(self, *args, **kwargs):
        self._user_kwargs = []
        for key, val in kwargs.items():
            if key not in ("ntime", "index"):
                self._user_kwargs.append(key)
                self._metadata.append(key)
                setattr(self, key, val)
        for key in self._user_kwargs:
            kwargs.pop(key)
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        def stats_post_process(result, **kwargs):
            return result

        self.stats_post_process = stats_post_process
        self._include_convenience_index()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.to_array(), index=self.index, **self.user_kwargs
        )

    @property
    def ntime(self):
        """Number of cadences in the data."""
        return self.shape[0]

    @classmethod
    def from_pandas(cls, data, **kwargs):
        """Convert a pd.Series to a DataSeries"""
        return cls(data, **kwargs)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return self.__class__(result, **self.user_kwargs)


class DataSeries(Series, StatsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_stats_methods()

    def __repr__(self):
        return f"📉 DataSeries {self.shape}\n" + super().__repr__()


class ErrorSeries(Series, ErrorStatsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_errstats_methods()

    def __repr__(self):
        return f"📈 ErrorSeries {self.shape}\n" + super().__repr__()


class BoolSeries(
    Series,
    BoolStatsMixin,
):
    def __repr__(self):
        return f"⚫️⚪️ BoolSeries {self.shape}\n" + super().__repr__()


class BitwiseSeries(
    Series,
    BitwiseMixin,
):
    _code_dict = None
    _values_display = None

    def __init__(self, *args, **kwargs):
        kwargs["codes"] = kwargs.get("codes", {})
        self.codes = kwargs["codes"]
        values_display = kwargs.pop("values_display", "bitwise")
        super().__init__(*args, **kwargs)
        self.values_display = values_display
        self._user_kwargs.append("values_display")

    def __repr__(self):
        if self._values_display == "detailed":
            display = self.apply(lambda x: self.parse_code(x)).__repr__()
        elif self._values_display == "parsed":
            display = self.apply(lambda x: self.breakdown(x)).__repr__()
        else:
            display = super().__repr__()
        return f"📗 BitwiseSeries {self.shape}\n" + display

    @property
    def codes(self):
        """Return the codes used in this BitwiseCube."""
        return self._code_dict

    @codes.setter
    def codes(self, codes_dict):
        self._code_dict = codes_dict

    @property
    def values_display(self):
        return self._values_display

    @values_display.setter
    def values_display(self, value):
        allowed = {"bitwise", "parsed", "detailed"}
        if value.lower() not in allowed:
            raise AttributeError(f"Display must be one of {allowed}.")
        self._values_display = value.lower()
