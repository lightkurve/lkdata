"""Classes and tools for working with 3 dimensional data."""
import logging
import pandas as pd
from pandas.io.formats.style import Styler
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
from .meta import CubeMeta

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
        exposure_time=None,
        **kwargs,
    ):
        self._metadata = []
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
            if row_indices == {}:
                row_indices["row"] = np.arange(nrow)
            if col_indices == {}:
                col_indices["column"] = np.arange(ncol)
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
        kwargs["exposure_time"] = exposure_time
        self._include_convenience_meta(**kwargs)
        self._repr_html_ = self._repr_html_

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
        """Create a stylized single cadence frame of a datacube"""
        cadence = int(np.floor(cadence))
        if isinstance(self.index, pd.MultiIndex):
            indices = []
            for i in zip(self.index.names, self.index[cadence]):
                if i[0] == "cadence":
                    strlabel = f"{i[0]}: {int(np.floor(i[1]))}"
                elif i[0] == "cadences":
                    strlabel = f"{i[0]}: {i[1]}"
                else:
                    strlabel = f"{i[0]}: {i[1]:0.3f}"
                indices += [strlabel]
        else:
            indices = [
                f"{i[0]} {i[1]}" for i in zip(self.index.names, [self.index[cadence]])
            ]
        str_index = "<br>" + "<br>".join(indices)
        row = self.__getattr__(self.columns.names[1])
        col = self.__getattr__(self.columns.names[2])
        df = pd.DataFrame(
            self.to_array()[cadence],
            index=pd.Series(row[:: self.ncol], name=self.columns.names[1]),
            columns=pd.MultiIndex.from_product(
                [[self.columns.names[2]], pd.Series(col[: self.ncol])]
            ),
        )
        out = Styler(df).set_caption(str_index)
        if self._stats_type == "error":
            out = out.format(precision=3)
        else:
            out = out.format(precision=0, thousands=",")

        out = out.background_gradient(
            axis=None,
            vmin=self.to_array()[cadence].min(),
            vmax=self.to_array()[cadence].max(),
            cmap="gray",
        )

        out = out.set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": "caption-side: bottom; font-size:1em; font-weight: bold;",
                },
                {"selector": "th", "props": "text-align: center;"},
                {
                    "selector": "td",
                    "props": "width: 30px; height: 30px; font-size: 6pt; text-align: center;",
                },
                {"selector": ":hover", "props": ""},
            ]
        )
        return out

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        out0 = self._single_cadence_frame(0)
        if self.shape[0] > 1:
            hidden_frames = f"[+{self.shape[0]-1} cadences]"
            return f"""
            {self.__repr__()}
            {out0.to_html(max_rows=10, max_columns=10)}
            ...<br>
            {hidden_frames}<br>
            """
        else:
            return f"""
            {self.__repr__()}
            {out0.to_html(max_rows=10, max_columns=10)}
            """

    @property
    def meta(self):
        return CubeMeta(self)

    @property
    def units(self):
        return self._flux_units

    @units.setter
    def units(self, unit):
        self._flux_units = str(unit)

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

    @classmethod
    def _build_instance(cls, new, **kwargs):
        return cls(new, ntime=len(new), **kwargs)

    @classmethod
    def _build_ds_instance(cls, new, **kwargs):
        return cls(new, ntime=len(new), **kwargs)

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

    def link_error(self, errorcube):
        self._metadata.append("error")
        self.error = errorcube
        return

    # @property
    # def loc(self):
    #     return self.loc

    # @loc.getter
    # def loc(self):
    #     idx = pd.IndexSlice
    #     return self.loc[:, idx]


class FluxCube:
    def __init__(self, data, error, **kwargs):
        self.data = DataCube(data, **kwargs)
        self.error = ErrorCube(error, **kwargs)
        self.kwargs = kwargs

    @staticmethod
    def applyfunc(obj, func, *args, **kwargs):
        func = getattr(obj, func)
        return func(*args, **kwargs)

    @classmethod
    def _build_fc_instance(cls, newdata, newerror, **kwargs):
        return cls(newdata, newerror, **kwargs)

    def __getattr__(self, attr, *args, **kwargs):
        if "_repr_" in attr:
            pass
        else:
            data_attr = getattr(self.data, attr)
            error_attr = getattr(self.error, attr)
            if callable(data_attr):

                def func(*args, **kwargs):
                    data_ret = data_attr(*args, **kwargs)
                    error_ret = error_attr(*args, **kwargs)
                    if isinstance(data_ret, self.data.__class__):
                        new = self._build_fc_instance(
                            data_ret.to_array(), error_ret.to_array(), **data_ret.meta
                        )
                        return new
                    else:
                        return (data_attr(*args, **kwargs), error_attr(*args, **kwargs))

                return func

            elif hasattr(data_attr, "__iter__"):
                if all(data_attr == error_attr):
                    return data_attr
                else:
                    return (data_attr, error_attr)
            elif data_attr == error_attr:
                return data_attr
            return (data_attr, error_attr)

    def __repr__(self):
        return f"{self.data.__repr__()}, {self.error.__repr__()}"

    def _repr_html_(self):
        return self.data._repr_html_().replace(
            self.data.__repr__(), self.data.__repr__() + ", " + self.error.__repr__()
        )


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
