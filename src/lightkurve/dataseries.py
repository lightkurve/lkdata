"""Classes and tools for working with 3 dimensional data."""

import logging
import pandas as pd

from .mixins import (
    StatsMixin,
    MathMixin,
    ErrorStatsMixin,
    PlotMixin,
    AggMixin,
    ConvenienceMixins,
)

log = logging.getLogger()


class DataSeries(
    StatsMixin, MathMixin, AggMixin, PlotMixin, ConvenienceMixins, pd.Series
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init__()

    def _lk_repr(self):
        return f"📉 DataSeries {self.shape}\n"

    def __repr__(self):
        return self._lk_repr() + super().__repr__()

    def __post_init__(self):
        def stats_post_process(result, **kwargs):
            return result

        self.stats_post_process = stats_post_process
        self._include_convenience_index()

    @property
    def ntime(self):
        return self.shape[0]

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return DataSeries(data, **kwargs)

    def _build_instance(self, new, **kwargs):
        return self.__class__(new, **kwargs)

    def to_array(self):
        return self.to_numpy()

    @property
    def _pd_class(self):
        return pd.Series


class ErrorSeries(ErrorStatsMixin, DataSeries):
    def _lk_repr(self):
        return f"📈 ErrorSeries {self.shape}\n"

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return ErrorSeries(data, **kwargs)
