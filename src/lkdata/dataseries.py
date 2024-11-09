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

    def __repr__(self):
        return self._lk_repr()

    def _repr_html_(self):
        return self._lk_repr() + "\n" + super().__repr__()

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

    def _lk_repr(self):
        return f"📉 DataSeries {self.shape}"


class ErrorSeries(Series, ErrorStatsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_errstats_methods()

    def _lk_repr(self):
        return f"📈 ErrorSeries {self.shape}"


class BoolSeries(
    Series,
    BoolStatsMixin,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"⚫️⚪️ BoolSeries {self.shape}"


class BitwiseSeries(
    Series,
    BitwiseMixin,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"📗 BitwiseSeries {self.shape}"
