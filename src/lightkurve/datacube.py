"""Classes and tools for working with 3 dimensional data."""

import pandas as pd
import numpy as np
import logging

from pandas._typing import Axis, AxisInt
from .mixins import StatsMixin, MathMixin, ErrorStatsMixin, PlotMixins

log = logging.getLogger()


class TimeSeries(StatsMixin, MathMixin, PlotMixins, pd.DataFrame):
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "time": 0, 1: 1, "pixel": 1}

    def __init__(self, data, *args, **kwargs):
        if data.ndim == 3:
            raise ValueError()
        elif data.ndim == 2:
            super().__init__(data, *args, **kwargs)
            self.ntime, self.npixel = data.shape
        else:
            raise ValueError()

        def make_pixelseries(result):
            log.debug("Modified result for pixelseries shape.")
            return result.to_numpy()

        def make_timeseries(result):
            log.debug("Modified result for timeseries shape.")
            return result.to_numpy()

        def stats_post_process(result, **kwargs):
            if kwargs.get("axis") in [0, "time"]:
                return make_pixelseries(result)
            elif kwargs.get("axis") in [1, "pixel"]:
                return make_timeseries(result)
            else:
                return result

        self.stats_post_process = stats_post_process

    def __repr__(self):
        return f"TimeSeries {self.ntime, self.npixel}"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        return self.__repr__()

    def _build_instance(self, new_data):
        return self.__class__(new_data)

    def aggregate(self, bins, aggregator="mean"):
        bin_edges = pd.cut(np.sort(self.index), bins, right=False)
        gb = self.groupby(bin_edges, observed=False)
        new_data = getattr(gb, aggregator)()
        return self.__class__.from_dataframe(
            new_data, index=bins[:-1] + np.diff(bins) / 2
        )


class DataCube(StatsMixin, MathMixin, PlotMixins, pd.DataFrame):
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "time": 0, 1: 1, "pixel": 1}

    def __init__(self, data, ntime=None, nrow=None, ncol=None, *args, **kwargs):
        if data.ndim == 3:
            super().__init__(np.vstack(data.transpose([1, 2, 0])).T, *args, **kwargs)
            self.ntime = data.shape[0] if ntime is None else ntime
            self.nrow = data.shape[1] if nrow is None else nrow
            self.ncol = data.shape[2] if ncol is None else ncol
        elif data.ndim == 2:
            if (nrow is None) | (ncol is None):
                raise ValueError("Must set `nrow` and `ncol`.")
            super().__init__(data, *args, **kwargs)
            self.ntime = data.shape[0] if ntime is None else ntime
            self.nrow, self.ncol = nrow, ncol
        else:
            raise ValueError()

        def make_image(result):
            log.debug("Modified result for image shape.")
            return result.to_numpy().reshape(self.nrow, self.ncol)

        def make_timeseries(result):
            log.debug("Modified result for timeseries shape.")
            return result.to_numpy()

        def stats_post_process(result, **kwargs):
            if kwargs.get("axis") in [0, "time"]:
                return make_image(result)
            elif kwargs.get("axis") in [1, "pixel"]:
                return make_timeseries(result)
            else:
                return result

        self.stats_post_process = stats_post_process

    @property
    def TimeSeries(self):
        return TimeSeries

    @property
    def npixel(self):
        return self.ncol * self.nrow

    def __getitem__(self, key):
        # Simple slice in time, results in DataCube
        if isinstance(key, (slice, np.ndarray, list, range)):
            return self.__class__.from_dataframe(
                self.iloc[key],
                nrow=self.nrow,
                ncol=self.ncol,
                index=self.index[key],
            )

        # Integer time, currently results in DataCube
        if isinstance(key, int):
            return self.__class__.from_dataframe(
                self.iloc[np.atleast_1d(key)],
                nrow=self.nrow,
                ncol=self.ncol,
                index=self.index[np.atleast_1d(key)],
            )

        if isinstance(key, tuple):
            time = key[0]
            if isinstance(key[0], (int, list, np.ndarray)):
                time = np.atleast_1d(time)
            elif isinstance(key[0], slice):
                time = range(self.ntime)[time]
            else:
                raise ValueError(f"Can not parse time {key[0]}")

            if len(key) == 1:
                return self.__class__.from_dataframe(
                    self.iloc[time],
                    nrow=self.nrow,
                    ncol=self.ncol,
                    index=self.index[time],
                )
            # If only two things passed
            if len(key) == 2:
                if np.ndim(key[1]) == 2:
                    aperture = key[1]
                    # Passed an aperture, needs to become time-series
                    return self[time].to_timeseries(
                        *np.where(aperture), index=self.index[time]
                    )
                else:
                    # If not, must be expecting a slice
                    row = key[1]
                    col = slice(self.ncol + 1)
            if len(key) == 3:
                row, col = key[1], key[2]

        # To be a a 3D dataset needs to pass slices or integers as row/column
        if not isinstance(row, (slice, int)) & isinstance(col, (slice, int)):
            return self[time].to_timeseries(row, col, index=self.index[time])
        nrow, ncol, df_index = self._convert_to_df_index(row, col)
        return self.__class__.from_dataframe(
            super().__getitem__(df_index).iloc[time],
            nrow=nrow,
            ncol=ncol,
            index=self.index[time],
        )

    def _convert_to_df_index(self, row, col):
        # Convert row, col index to DataFrame column index
        if isinstance(row, slice):
            row_indices = range(self.nrow)[row]
            nrow = len(range(*row.indices(self.nrow)))
        elif isinstance(row, range):
            row_indices = list(row)
        elif isinstance(row, int):
            row_indices = [row]
            nrow = 1
        else:
            row_indices = row
            nrow = len(row)

        if isinstance(col, slice):
            col_indices = range(self.ncol)[col]
            ncol = len(range(*col.indices(self.ncol)))
        elif isinstance(col, range):
            col_indices = list(col)
        elif isinstance(col, int):
            col_indices = [col]
            ncol = 1
        else:
            col_indices = col
            ncol = len(col)
        return nrow, ncol, [r * self.ncol + c for r in row_indices for c in col_indices]

    def __repr__(self):
        return f"DataCube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        return self.__repr__()

    @staticmethod
    def from_dataframe(data, nrow, ncol, index=None):
        """Convert a pd.DataFrame to a DataCube"""
        return DataCube(
            data.to_numpy(), ntime=len(data), nrow=nrow, ncol=ncol, index=index
        )

    def to_dataframe(self):
        return pd.DataFrame(self.to_numpy(), index=self.index)

    def to_timeseries(self, row, col, index=None):
        if isinstance(row, slice):
            row_indices = range(self.nrow)[row]
        else:
            row_indices = np.atleast_1d(row)
        if isinstance(col, slice):
            col_indices = range(self.ncol)[col]
        else:
            col_indices = np.atleast_1d(col)
        df_index = np.asarray(row_indices) * self.ncol + np.asarray(col_indices)
        return self.TimeSeries(super().__getitem__(df_index).to_numpy(), index=index)

    @property
    def cube_shape(self):
        return (self.ntime, self.nrow, self.ncol)

    def asarray(self):
        return (
            self.to_numpy()
            .T.reshape((self.nrow, self.ncol, self.ntime))
            .transpose([2, 0, 1])
        )

    def _build_instance(self, new_data):
        return self.__class__(
            new_data, ntime=self.ntime, nrow=self.nrow, ncol=self.ncol
        )

    def get_frame(self, frame_number):
        return self[frame_number].to_numpy().reshape(self.nrow, self.ncol)

    def aggregate(self, bins, aggregator="mean"):
        bin_edges = pd.cut(np.sort(self.index), bins, right=False)
        gb = self.groupby(bin_edges, observed=False)
        new_data = getattr(gb, aggregator)()
        return self.__class__.from_dataframe(
            new_data,
            nrow=self.nrow,
            ncol=self.ncol,
            index=bins[:-1] + np.diff(bins) / 2,
        )

    def downsample(self, nframes):
        dt = nframes * np.median(np.diff(self.index))
        bins = np.arange(self.index[0], self.index[-1] + 1 * dt, dt)
        bin_edges = pd.cut(np.sort(self.index), bins, right=False)
        gb = self.groupby(bin_edges, observed=False)

        new = gb.sum()
        count = gb[0].count()
        k = np.asarray(count == nframes)[:, 0]
        return self.__class__.from_dataframe(
            new[k],
            nrow=self.nrow,
            ncol=self.ncol,
            index=(bins[:-1] + np.diff(bins) / 2)[k],
        )


class ErrorTimeSeries(ErrorStatsMixin, TimeSeries, MathMixin):
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "time": 0, 1: 1, "pixel": 1}

    def __repr__(self):
        return f"ErrorTimeSeries {self.ntime, self.npixel}"

    def __add__(self, val):
        return self._build_instance(
            (self.to_numpy() ** 2 + self._process_val(val) ** 2) ** 0.5
        )


class ErrorCube(ErrorStatsMixin, DataCube, MathMixin):
    _AXIS_TO_AXIS_NUMBER: dict[Axis, AxisInt] = {0: 0, "time": 0, 1: 1, "pixel": 1}

    def __repr__(self):
        return f"ErrorCube {self.ntime, self.nrow, self.ncol}"

    def __add__(self, val):
        return self._build_instance(
            (self.to_numpy() ** 2 + self._process_val(val) ** 2) ** 0.5
        )

    @property
    def TimeSeries(self):
        return ErrorTimeSeries

    @staticmethod
    def from_dataframe(data, nrow, ncol, index=None):
        """Convert a pd.DataFrame to a DataCube"""
        return ErrorCube(
            data.to_numpy(), ntime=len(data), nrow=nrow, ncol=ncol, index=index
        )

    def aggregate(self, bins, aggregator="mean"):
        bin_edges = pd.cut(np.sort(self.index), bins, right=False)
        if aggregator in ["mean", "median", "sum"]:
            gb = (self**2).groupby(bin_edges, observed=False)
            new_data = (gb).sum() ** 0.5
            if aggregator != "sum":
                count = gb[0].count()
                new_data /= count
        elif aggregator == "std":
            gb = self.groupby(bin_edges, observed=False)
            new_data = gb.median()
            count = gb[0].count()
            new_data /= np.sqrt(count * 2)
        elif aggregator == "count":
            gb = self.groupby(bin_edges, observed=False)
            new_data = gb.count()
        else:
            raise ValueError(f"No aggregator {aggregator}.")

        return self.__class__.from_dataframe(
            new_data,
            nrow=self.nrow,
            ncol=self.ncol,
            index=bins[:-1] + np.diff(bins) / 2,
        )

    def downsample(self, nframes):
        dt = nframes * np.median(np.diff(self.index))
        bins = np.arange(self.index[0], self.index[-1] + 1 * dt, dt)
        bin_edges = pd.cut(np.sort(self.index), bins, right=False)
        gb = (self**2).groupby(bin_edges, observed=False)
        new = gb.sum() ** 0.5
        count = gb[0].count()
        k = np.asarray(count == nframes)[:, 0]
        return self.__class__.from_dataframe(
            new[k],
            nrow=self.nrow,
            ncol=self.ncol,
            index=(bins[:-1] + np.diff(bins) / 2)[k],
        )
