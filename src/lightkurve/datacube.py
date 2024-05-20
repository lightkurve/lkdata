"""Classes and tools for working with 3 dimensional data."""
import logging
import pandas as pd
import numpy as np

from .dataframe import DataFrame, ErrorFrame
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


class DataCube(
    StatsMixin, MathMixin, AggMixin, PlotMixin, ConvenienceMixins, pd.DataFrame
):
    def __init__(
        self,
        data,
        ntime=None,
        nrow=None,
        ncol=None,
        time_indices={},
        row_indices={},
        col_indices={},
        index=None,
        columns=None,
    ):
        if data.ndim == 2:
            if (nrow is None) | (ncol is None):
                raise ValueError("Must set `nrow` and `ncol`.")
        elif data.ndim == 3:
            ntime = data.shape[0] if ntime is None else ntime
            nrow = data.shape[1] if nrow is None else nrow
            ncol = data.shape[2] if ncol is None else ncol
            data = np.vstack(data.transpose([1, 2, 0])).T
        else:
            raise ValueError()

        if index is None:
            index = pd.MultiIndex.from_arrays(
                [np.arange(ntime), *list(time_indices.values())],
                names=["cadence", *list(time_indices.keys())],
            )
        if columns is None:
            columns = pd.MultiIndex.from_arrays(
                [
                    np.arange(nrow * ncol).ravel(),
                    *[
                        (
                            value[:, None] * np.ones((nrow, ncol), dtype=value.dtype)
                        ).ravel()
                        for value in row_indices.values()
                    ],
                    *[
                        (value * np.ones((nrow, ncol), dtype=value.dtype)).ravel()
                        for value in col_indices.values()
                    ],
                ],
                names=["series", *list(row_indices.keys()), *list(col_indices.keys())],
            )

        super().__init__(data, index=index, columns=columns)
        self.ntime, self.nrow, self.ncol = ntime, nrow, ncol

        def make_image(result):
            # log.debug("Modified result for image shape.")
            return result.to_numpy().reshape(self.nrow, self.ncol)

        def make_timeseries(result):
            # log.debug("Modified result for timeseries shape.")
            return self._series_class(result)

        def stats_post_process(result, **kwargs):
            if kwargs.get("axis") in [0, "time"]:
                return make_image(result)
            elif kwargs.get("axis") in [1, "series"]:
                return make_timeseries(result)
            else:
                return result

        self.stats_post_process = stats_post_process
        self._include_convenience_index()
        self._include_convenience_columns()

    @property
    def nseries(self):
        return self.ncol * self.nrow

    def __getitem__(self, key):
        # Simple slice in time, results in DataCube
        if isinstance(key, (slice, np.ndarray, list, range)):
            return self.__class__.from_pandas(
                self.iloc[key],
                nrow=self.nrow,
                ncol=self.ncol,
                index=self.index[key],
                columns=self.columns,
            )

        # Integer time, currently results in DataCube
        if isinstance(key, int):
            return self.__class__.from_pandas(
                self.iloc[np.atleast_1d(key)],
                nrow=self.nrow,
                ncol=self.ncol,
                index=self.index[np.atleast_1d(key)],
                columns=self.columns,
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
                return self.__class__.from_pandas(
                    self.iloc[time],
                    nrow=self.nrow,
                    ncol=self.ncol,
                    index=self.index[time],
                    columns=self.columns,
                )
            # If only two things passed
            if len(key) == 2:
                if np.ndim(key[1]) == 2:
                    aperture = key[1]
                    # Passed an aperture, needs to become time-series
                    return self[time].to_dataframe(*np.where(aperture))
                else:
                    # If not, must be expecting a slice
                    row = key[1]
                    col = slice(self.ncol + 1)
            if len(key) == 3:
                row, col = key[1], key[2]

        # To be a a 3D dataset needs to pass slices or integers as row/column
        if not isinstance(row, (slice)) & isinstance(col, (slice)):
            return self[time].to_dataframe(row, col)
        nrow, ncol, series_index = self._convert_to_series_index(row, col)
        return self.__class__.from_pandas(
            self.iloc[time, series_index],
            nrow=nrow,
            ncol=ncol,
            index=self.index[time],
            columns=self.columns[series_index],
        )

    def _convert_to_series_index(self, row, col):
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

    def _single_cadence_frame(self, cadence):
        df = pd.DataFrame(
            self.to_array()[cadence],
            index=self.row[:: self.nrow],
            columns=self.column[: self.ncol],
        )
        df.index.name = "row"
        df.columns.name = "column"
        return df

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        out = "\n"
        with np.printoptions(linewidth=79, edgeitems=2, threshold=100):
            for time_index in self.index.names:
                out += f"{time_index.ljust(16)}:\t{self.__getattr__(time_index)}\n"
            for loc in self.columns.names:
                if loc != "series":
                    out += f"{loc.ljust(16)}:\t{np.unique(self.__getattr__(loc))}\n"
            out += "type".ljust(16) + f":\t{self.__class__}\n"
            return print(self.__repr__(), "\n", out)

    @staticmethod
    def from_pandas(data, nrow, ncol, **kwargs):
        """Convert a pd.DataFrame to a DataCube"""
        return DataCube(
            data.to_numpy(), ntime=len(data), nrow=nrow, ncol=ncol, **kwargs
        )

    def to_dataframe(self, row, col, **kwargs):
        if isinstance(row, slice):
            row_indices = range(self.nrow)[row]
        else:
            row_indices = np.atleast_1d(row)
        if isinstance(col, slice):
            col_indices = range(self.ncol)[col]
        else:
            col_indices = np.atleast_1d(col)
        series_index = np.asarray(row_indices) * self.ncol + np.asarray(col_indices)
        return self._frame_class(
            self.iloc[:, series_index],
            index=self.index,
            columns=self.columns[series_index],
        )

    def _build_instance(self, new, **kwargs):
        return self.__class__(
            new, ntime=len(new), nrow=self.nrow, ncol=self.ncol, **kwargs
        )

    def _build_ds_instance(self, new, **kwargs):
        return self.__class__(new, ntime=len(new), **kwargs)

    def to_array(self):
        return self.to_numpy().reshape(self.ntime, self.nrow, self.ncol)

    @property
    def _frame_class(self):
        return DataFrame

    @property
    def _series_class(self):
        return DataSeries

    @property
    def _pd_class(self):
        return pd.DataFrame


class ErrorCube(ErrorStatsMixin, DataCube):
    def __repr__(self):
        return f"📕 ErrorCube {self.ntime, self.nrow, self.ncol}"

    @staticmethod
    def from_pandas(data, nrow, ncol, **kwargs):
        """Convert a pd.DataFrame to a DataCube"""
        return ErrorCube(
            data.to_numpy(), ntime=len(data), nrow=nrow, ncol=ncol, **kwargs
        )

    @property
    def _frame_class(self):
        return ErrorFrame

    @property
    def _series_class(self):
        return ErrorSeries
