"""Classes and tools for working with 3 dimensional data."""
from functools import singledispatchmethod
from collections.abc import Iterable
import logging
import numpy as np
import pandas as pd

from .dataseries import DataSeries, ErrorSeries
from .mixins import (
    StatsMixin,
    MathMixin,
    ErrorStatsMixin,
    PlotMixin,
    AggMixin,
    ConvenienceMixins,
)

log = logging.getLogger()


class DataFrame(
    StatsMixin, MathMixin, AggMixin, PlotMixin, ConvenienceMixins, pd.DataFrame
):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.__post_init__()

    @property
    def nseries(self):
        return self.shape[1]

    def __repr__(self):
        return f"🟦 DataFrame {self.shape}"

    def _repr_html_(self):
        return self.__repr__() + super()._repr_html_()

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
        return self.shape[0]

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return DataFrame(data, **kwargs)

    def _build_instance(self, new, **kwargs):
        return self.__class__(new, **kwargs)

    def to_array(self):
        return self.to_numpy()

    @property
    def _series_class(self):
        return DataSeries

    @property
    def _pd_class(self):
        return pd.DataFrame

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register
    def _(self, key: int | Iterable | slice):
        return self.__class__.from_pandas(
            self.iloc[key], index=self.index[key], columns=self.columns
        )

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        if isinstance(key[1], slice):
            series_index = np.arange(self.nseries)[key[1]]
        elif isinstance(key[1], Iterable):
            series_index = key[1]

        return self.__class__.from_pandas(
            self.iloc[time_key, series_index],
            index=self.index[time_key],
            columns=self.columns[series_index],
        )


class ErrorFrame(ErrorStatsMixin, DataFrame):
    def __repr__(self):
        return f"🟥 ErrorFrame {self.shape}"

    @property
    def _series_class(self):
        return ErrorSeries

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return ErrorFrame(data, **kwargs)
