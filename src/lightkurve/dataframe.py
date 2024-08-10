"""Classes and tools for working with 3 dimensional data."""

import logging
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
        super().__init__()
        self.__post_init__()

    @property
    def nseries(self):
        return self.shape[1]

    def __repr__(self):
        return f"🟦 DataFrame {self.shape}\n"

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


class ErrorFrame(ErrorStatsMixin, DataFrame):
    def __repr__(self):
        return f"🟥 ErrorFrame {self.shape}\n"

    @property
    def _series_class(self):
        return ErrorSeries

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return ErrorFrame(data, **kwargs)
