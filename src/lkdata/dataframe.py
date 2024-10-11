"""Classes and tools for working with 3 dimensional data."""
import logging
from abc import ABC
from collections.abc import Iterable
from functools import singledispatchmethod

import numpy as np
import pandas as pd

from .dataseries import DataSeries, ErrorSeries
from .mixins import (
    AggMixin,
    ConvenienceMixins,
    ErrorStatsMixin,
    MathMixin,
    StatsMixin,
)

log = logging.getLogger()


class Frame(
    ABC,
    MathMixin,
    AggMixin,
    ConvenienceMixins,
    pd.DataFrame,
):
    """Abstract dataclass for frame-like data with time and multiple series"""

    _pd_class = pd.DataFrame
    row_names = None
    col_names = None
    _user_kwargs = None

    def __init__(self, *args, **kwargs):
        index = kwargs.pop("index", None)
        time_indices = kwargs.pop("time_indices", None)
        index = self.parse_index(index, time_indices)
        if index.empty:
            index = None

        for key, val in kwargs.items():
            if key not in (
                "ntime",
                "index",
                "columns",
                "time_indices",
                "row_indices",
                "col_indices",
            ):
                self._metadata.append(key)
                setattr(self, key, val)
                if key not in ("nrow", "ncol"):
                    self._user_kwargs.append(key)

        for key in self._metadata:
            kwargs.pop(key, None)

        pd.DataFrame.__init__(self, *args, index=index, **kwargs)
        self.__post_init__()

    @property
    def nseries(self):
        """Number of series in the DataFrame"""
        return self.shape[1]

    def _repr_html_(self):
        return repr(self) + super()._repr_html_()

    def __post_init__(self):
        def make_pixelseries(result):
            # log.debug("Modified result for pixelseries shape.")
            return self._series_class.from_pandas(result)

        def make_timeseries(result):
            # log.debug("Modified result for timeseries shape.")
            return self._series_class.from_pandas(result)

        def stats_post_process(result, **kwargs):
            if kwargs.get("axis") in [0, "time"]:
                return make_pixelseries(result)
            elif kwargs.get("axis") in [1, "pixel"]:
                return make_timeseries(result)
            else:
                return result

        self.stats_post_process = stats_post_process
        self._include_convenience_index()
        self._include_convenience_columns()

    @property
    def ntime(self):
        """Number of time frames"""
        return self.shape[0]

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register(Iterable)
    @__getitem__.register(slice)
    def _(self, key):
        return self.__class__.from_pandas(
            self.iloc[key],
            index=self.index[key],
            columns=self.columns,
            **self.user_kwargs,
        )

    @__getitem__.register(int)
    def _(self, key):
        # bypassing pandas conversions to series when given int keys
        return self[key : key + 1]

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        if isinstance(key[1], slice):
            series_index = np.arange(self.nseries)[key[1]]
        elif isinstance(key[1], Iterable):
            series_index = key[1]
        else:
            return self._series_class(
                self.iloc[time_key, key[1]],
                index=self.index[time_key],
                **self.user_kwargs,
            )

        return self.__class__.from_pandas(
            self.iloc[time_key, series_index],
            index=self.index[time_key],
            columns=self.columns[series_index],
            **self.user_kwargs,
        )

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.to_array(), index=self.index, columns=self.columns, **self.user_kwargs
        )


class DataFrame(Frame, StatsMixin):
    _series_class = DataSeries

    def __init__(self, *args, **kwargs):
        self._metadata = []
        self._user_kwargs = []
        super().__init__(*args, **kwargs)
        self._set_stats_methods()

    def __repr__(self):
        return f"🟦 DataFrame {self.shape}"

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return DataFrame(data, **kwargs)


class ErrorFrame(Frame, ErrorStatsMixin):
    _series_class = ErrorSeries

    def __init__(self, *args, **kwargs):
        self._metadata = []
        self._user_kwargs = []
        super().__init__(*args, **kwargs)
        self._set_errstats_methods()

    def __repr__(self):
        return f"🟥 ErrorFrame {self.shape}"

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return ErrorFrame(data, **kwargs)
