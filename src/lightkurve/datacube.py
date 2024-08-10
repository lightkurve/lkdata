"""Classes and tools for working with 3 dimensional data."""
import logging
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from abc import ABC

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


class Cube(ABC, pd.DataFrame):
    """Abstract dataclass for cube-like data with time, row, and column axes"""

    ntime = None
    nrow = None
    ncol = None
    row_names = None
    col_names = None

    def __init__(
        self,
        data: np.ndarray,
        time_indices: dict = None,
        row_indices: dict = None,
        col_indices: dict = None,
        **kwargs,
    ):
        data = self._preprocess_data(data)
        if time_indices is None:
            time_indices = {"time_index": np.arange(self.ntime)}
        if "time_index" in time_indices.keys():
            arrays = [*list(time_indices.values())]
            names = [*list(time_indices.keys())]
        else:
            arrays = [np.arange(self.ntime), *list(time_indices.values())]
            names = ["time_index", *list(time_indices.keys())]
        index = pd.MultiIndex.from_arrays(arrays, names=names)
        if row_indices is None:
            row_indices = {"row": np.arange(self.nrow)}
        if col_indices is None:
            col_indices = {"col": np.arange(self.ncol)}
        self.row_names = list(row_indices.keys())
        self.col_names = list(col_indices.keys())
        columns = pd.MultiIndex.from_arrays(
            arrays=[
                np.arange(self.nrow * self.ncol).ravel(),
                *[
                    (
                        value[:, None]
                        * np.ones((self.nrow, self.ncol), dtype=value.dtype)
                    ).ravel()
                    for value in row_indices.values()
                ],
                *[
                    (value * np.ones((self.nrow, self.ncol), dtype=value.dtype)).ravel()
                    for value in col_indices.values()
                ],
            ],
            names=[
                "series",
                *list(row_indices.keys()),
                *list(col_indices.keys()),
            ],
        )

        assert data.shape[0] == self.ntime, "data shape doesn't match given time"
        assert (
            data.shape[1] == self.nrow * self.ncol
        ), "data shape doesn't match given rows and columns"
        super().__init__(data, index=index, columns=columns)

    def _preprocess_data(self, data):
        data = np.array(data)
        if data.ndim == 2:
            if (self.nrow is None) | (self.ncol is None):
                raise ValueError(
                    """
                Must set `nrow` and `ncol` when giving data as a 2D array.
                """
                )
        elif data.ndim == 3:
            self._set_dim("ntime", data.shape[0])
            self._set_dim("nrow", data.shape[1])
            self._set_dim("ncol", data.shape[2])
            data = np.hstack(data.transpose([1, 0, 2]))
        else:
            raise ValueError()
        return data

    def _set_dim(self, attr, val):
        if getattr(self, attr, None) is None:
            setattr(self, attr, val)
        elif getattr(self, attr) != val:
            raise ValueError(f"Given {attr} does not match given data shape {val}")


class DataCube(
    Cube,
    StatsMixin,
    MathMixin,
    AggMixin,
    PlotMixin,
    ConvenienceMixins,
):
    def __init__(
        self,
        data,
        ntime: int = None,
        nrow: int = None,
        ncol: int = None,
        time_indices=None,
        row_indices=None,
        col_indices=None,
        index=None,
        columns=None,
        exposure_time=None,
        **kwargs,
    ):
        self._metadata = []
        self.ntime, self.nrow, self.ncol = ntime, nrow, ncol

        # prefer existing index and columns, if available
        if columns is not None:
            if row_indices is not None:
                self.row_names = list(row_indices.keys())
            if col_indices is not None:
                self.col_names = list(col_indices.keys())
            row_indices, col_indices = self._parse_columns(columns)
        if index is not None:
            time_indices = self._parse_index(index)

        super().__init__(data, time_indices, row_indices, col_indices)

        self._include_convenience_index()
        self._include_convenience_columns()
        self._include_convenience_meta(exposure_time=exposure_time, **kwargs)

    @staticmethod
    def _parse_index(index):
        time_names = index.names
        time_indices = {name: index.get_level_values(name) for name in time_names}
        return time_indices

    def _parse_columns(self, columns):
        row_names = getattr(self, "row_names", []) or []
        col_names = getattr(self, "col_names", []) or []

        for name in columns.names:
            if ("row" in name) and (name not in row_names):
                row_names.append(name)
            elif ("col" in name) and (name not in col_names):
                col_names.append(name)
        if (len(row_names) == 0) or (len(col_names) == 0):
            raise ValueError(
                """
            row and column indices cannot be inferred from a pandas.MultiIndex,
            specify the rows and columns in the row_indices and col_indices dicts.
            """
            )
        self.row_names = row_names
        self.col_names = col_names
        row_indices = {
            name: np.unique(columns.get_level_values(name)) for name in row_names
        }
        col_indices = {
            name: np.unique(columns.get_level_values(name)) for name in col_names
        }
        return row_indices, col_indices

    def stats_post_process(self, result, **kwargs):
        if kwargs.get("axis") in [0, "time"]:
            return result.to_numpy().reshape(self.nrow, self.ncol)
        elif kwargs.get("axis") in [1, "series"]:
            return self._series_class(result)
        else:
            return result

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
                row_indices={name: None for name in self.row_names},
                col_indices={name: None for name in self.col_names},
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
                if i[0] == "cadences":
                    strlabel = f"{i[0]}: {i[1]}"
                elif "cadence" in i[0]:
                    strlabel = f"{i[0]}: {int(np.floor(i[1]))}"
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
            {repr(self)}
            {out0.to_html(max_rows=11, max_columns=11)}
            ...<br>
            {hidden_frames}<br>
            """
        else:
            return f"""
            {repr(self)}
            {out0.to_html(max_rows=11, max_columns=11)}
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


class MaskedCube(DataCube):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
