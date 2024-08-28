"""Classes and tools for working with 3 dimensional data."""
import logging
from abc import ABC
from functools import singledispatchmethod
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


class Cube(ABC, pd.DataFrame, MathMixin, PlotMixin, AggMixin, ConvenienceMixins):
    """Abstract dataclass for cube-like data with time, row, and column axes"""

    ntime = None
    nrow = None
    ncol = None
    row_names = None
    col_names = None

    def __init__(
        self,
        data: list | np.ndarray,
        time_indices: dict | list = None,
        row_indices: dict | list = None,
        col_indices: dict | list = None,
        **kwargs,
    ):
        self._metadata = []

        self.ntime = kwargs.get("ntime", None)
        self.nrow = kwargs.get("nrow", None)
        self.ncol = kwargs.get("ncol", None)
        index = kwargs.get("index", None)
        columns = kwargs.get("columns", None)
        data = self._preprocess_data(data)

        index = self._parse_index(index, time_indices)
        columns = self._parse_columns(columns, row_indices, col_indices)

        super().__init__(data, index=index, columns=columns)
        self._include_convenience_index()
        self._include_convenience_columns()
        self._include_convenience_meta(**kwargs)

    def _preprocess_data(self, data):
        data = np.array(data)
        log.info("data.ndim = %s, data.shape= %s", data.ndim, data.shape)
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
            raise ValueError("""Dimension of given data not interpretable as a cube""")
        return data

    def _set_dim(self, attr, val):
        if getattr(self, attr, None) is None:
            setattr(self, attr, val)
        elif getattr(self, attr) != val:
            raise ValueError(f"Given {attr} does not match given data shape {val}")

    def _parse_index(self, index: pd.MultiIndex, time_indices: dict):
        """Retrieve time_indices from a pd.MultiIndex"""

        if index is not None:
            # prefer existing index, if available
            time_names = index.names
            time_indices = {name: index.get_level_values(name) for name in time_names}
        elif not time_indices:
            time_indices = {"time_index": np.arange(self.ntime)}

        if "time_index" in time_indices.keys():
            arrays = [*list(time_indices.values())]
            names = [*list(time_indices.keys())]
        else:
            arrays = [np.arange(self.ntime), *list(time_indices.values())]
            names = ["time_index", *list(time_indices.keys())]

        index = pd.MultiIndex.from_arrays(arrays, names=names)
        return index

    def _parse_columns(
        self,
        columns: pd.MultiIndex,
        row_indices: dict,
        col_indices: dict,
    ):
        # prefer existing columns, if available
        if columns is not None:
            if row_indices:
                self.row_names = list(row_indices.keys())
            if col_indices:
                self.col_names = list(col_indices.keys())

            row_names = getattr(self, "row_names") or []
            col_names = getattr(self, "col_names") or []

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

            row_indices = {
                name: np.unique(columns.get_level_values(name)) for name in row_names
            }
            col_indices = {
                name: np.unique(columns.get_level_values(name)) for name in col_names
            }

        if not row_indices:
            row_indices = {"row": np.arange(self.nrow)}
        if not col_indices:
            col_indices = {"col": np.arange(self.ncol)}

        self.row_names = list(row_indices.keys())
        self.col_names = list(col_indices.keys())

        def flatten(value):
            """Flatten row and column arrays"""
            return (value * np.ones((self.nrow, self.ncol), dtype=value.dtype)).ravel()

        row_arrs = [flatten(value[:, None]) for value in row_indices.values()]
        col_arrs = [flatten(value) for value in col_indices.values()]

        columns = pd.MultiIndex.from_arrays(
            arrays=[np.arange(self.nrow * self.ncol).ravel(), *row_arrs, *col_arrs],
            names=["series", *list(row_indices.keys()), *list(col_indices.keys())],
        )

        return columns

    def _convert_to_series_index(self, row, col):
        # Convert row, col index to DataFrame column index
        if isinstance(row, slice):
            row_indices = np.arange(self.nrow)[row]
        else:
            row_indices = np.atleast_1d(row)
        if isinstance(col, slice):
            col_indices = np.arange(self.ncol)[col]
        else:
            col_indices = np.atleast_1d(col)
        nrow = len(row_indices)
        ncol = len(col_indices)
        series_index = (
            row_indices.repeat(ncol).reshape(nrow, ncol).T * self.ncol
            + col_indices.reshape(ncol, 1)
        ).ravel()
        series_index.sort()
        return nrow, ncol, series_index

    def single_frame(self, cadence: int) -> Styler:
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
        row = getattr(self, self.columns.names[1])
        col = getattr(self, self.columns.names[2])
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

    @classmethod
    def _build_instance(cls, new, **kwargs):
        return cls(new, ntime=len(new), **kwargs)

    def _repr_html_(self):
        if hasattr(self, "_styler"):
            out0 = self.styler
        else:
            out0 = self.single_frame(0)
            self.styler = out0

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

    def __repr__(self):
        return f"Cube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register
    def _(self, key: slice | np.ndarray | list | range):
        # Simple slice in time, results in DataCube
        return self.__class__.from_pandas(
            self.iloc[key],
            nrow=self.nrow,
            ncol=self.ncol,
            index=self.index[key],
            columns=self.columns,
            row_indices={name: None for name in self.row_names},
            col_indices={name: None for name in self.col_names},
        )

    @__getitem__.register
    def _(self, key: int):
        # Integer time, currently results in DataCube
        return self.__class__.from_pandas(
            self.iloc[np.atleast_1d(key)],
            nrow=self.nrow,
            ncol=self.ncol,
            index=self.index[np.atleast_1d(key)],
            columns=self.columns,
        )

    @__getitem__.register
    def _(self, key: tuple):
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
        if not isinstance(row, (slice)) & (not isinstance(col, (slice))):
            return self[time].to_dataframe(row, col)
        nrow, ncol, series_index = self._convert_to_series_index(row, col)
        return self.__class__.from_pandas(
            self.iloc[time, series_index],
            nrow=nrow,
            ncol=ncol,
            index=self.index[time],
            columns=self.columns[series_index],
        )

    @property
    def meta(self):
        return CubeMeta(self)

    @property
    def nseries(self):
        return self.ncol * self.nrow

    @property
    def units(self):
        return self._flux_units

    @units.setter
    def units(self, unit):
        # remove formatting, if present
        self._flux_units = str(unit)

    @property
    def styler(self):
        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, input: Styler):
        self._styler = input

    def stats_post_process(self, result, **kwargs):
        if kwargs.get("axis") in [0, "time"]:
            return result.to_numpy().reshape(self.nrow, self.ncol)
        elif kwargs.get("axis") in [1, "series"]:
            return self._series_class(result)
        else:
            return result

    def to_dataframe(
        self, row: int | float | list | slice, col: int | float | list | slice, **kwargs
    ) -> DataFrame | ErrorFrame:
        """Convert Cube to Frame with the given row and column indices.

        Parameters
        ----------
        row : int | float | list | slice
            Index/list of indices or slice of row indices to include.
        col : int | float | list | slice
            Index/list of indices or slice of column indices to include.

        Returns
        -------
        DataFrame | ErrorFrame
            A Frame object of the same type as the input data, either
            DataFrame or ErrorFrame.
        """
        _, _, series_index = self._convert_to_series_index(row, col)
        return self._frame_class(
            self.iloc[:, series_index],
            index=self.index,
            columns=self.columns[series_index],
            **kwargs,
        )

    def to_array(self) -> np.ndarray:
        """Convert Cube data to a numpy array

        Returns
        -------
        data_array
            np.ndarray
        """
        data_array = self.to_numpy().reshape(self.ntime, self.nrow, self.ncol)
        return data_array

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, nrow: int, ncol: int, **kwargs):
        """Convert a pd.DataFrame to a DataCube

        Notes:
        This assumes no multi-indexing in the pandas dataframe.
        """
        return cls(data.to_numpy(), ntime=len(data), nrow=nrow, ncol=ncol, **kwargs)


class DataCube(
    Cube,
    StatsMixin,
):
    _frame_class = DataFrame
    _series_class = DataSeries
    _pd_class = pd.DataFrame

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}"


class ErrorCube(
    Cube,
    ErrorStatsMixin,
):
    _frame_class = ErrorFrame
    _series_class = ErrorSeries
    _pd_class = pd.DataFrame

    def __repr__(self):
        return f"📕 ErrorCube {self.ntime, self.nrow, self.ncol}"


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
