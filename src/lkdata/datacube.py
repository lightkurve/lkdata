"""Classes and tools for working with 3 dimensional data."""
import logging
from abc import ABC
from functools import singledispatchmethod
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from typing import Union, List, Dict, Optional

from .dataframe import DataFrame, ErrorFrame
from .dataseries import DataSeries, ErrorSeries
from .mixins import (
    StatsMixin,
    MathMixin,
    ErrorStatsMixin,
    AggMixin,
    ConvenienceMixins,
)

log = logging.getLogger()


class Cube(
    ABC,
    MathMixin,
    AggMixin,
    ConvenienceMixins,
    pd.DataFrame,
):
    """Abstract dataclass for cube-like data with time, row, and column axes"""

    ntime: Optional[int] = None
    nrow: Optional[int] = None
    ncol: Optional[int] = None
    row_names: Optional[List[str]] = None
    col_names: Optional[List[str]] = None
    _user_kwargs: Optional[List[str]] = None

    def __init__(
        self,
        data: Union[List, np.ndarray],
        time_indices: Union[Dict, List, None] = None,
        row_indices: Union[Dict, List, None] = None,
        col_indices: Union[Dict, List, None] = None,
        **kwargs,
    ):
        self.nrow = kwargs.get("nrow", None)
        self.ncol = kwargs.get("ncol", None)
        index = kwargs.get("index", None)
        columns = kwargs.get("columns", None)

        for key, val in kwargs.items():
            if key not in ("ntime", "nrow", "ncol", "index", "columns"):
                self._user_kwargs.append(key)
                self._metadata.append(key)
                setattr(self, key, val)

        data = self._preprocess_data(data)
        index = self.parse_index(index, time_indices, self.ntime)
        columns = self._parse_columns(columns, row_indices, col_indices)

        super().__init__(data, index=index, columns=columns)
        self._include_convenience_index()
        self._include_convenience_columns()

    def _preprocess_data(self, data):
        data = np.array(data)
        self._set_dim("ntime", data.shape[0])
        log.info("data.ndim = %s, data.shape= %s", data.ndim, data.shape)
        if data.ndim == 2:
            if (self.nrow is None) | (self.ncol is None):
                raise ValueError(
                    """
                Must set `nrow` and `ncol` when giving data as a 2D array.
                """
                )
        elif data.ndim == 3:
            self._set_dim("nrow", data.shape[1])
            self._set_dim("ncol", data.shape[2])
            data = np.hstack(data.transpose([1, 0, 2]))
        else:
            raise ValueError("""Dimension of given data not interpretable as a Cube""")
        return data

    def _set_dim(self, attr, val):
        if getattr(self, attr, None) is None:
            setattr(self, attr, val)
        elif getattr(self, attr) != val:
            raise ValueError(f"Given {attr} does not match given data shape {val}")

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
        if isinstance(row, slice) or isinstance(col, slice):
            series_index = (
                row_indices.repeat(ncol).reshape(nrow, ncol).T * self.ncol
                + col_indices.reshape(ncol, 1)
            ).ravel()
        else:
            series_index = row_indices * self.ncol + col_indices
        series_index.sort()
        return nrow, ncol, series_index

    def single_frame(self, cadence: int) -> Styler:
        """Create a stylized single cadence frame of a datacube"""
        cadence = int(np.floor(cadence))
        if isinstance(self.index, pd.MultiIndex):
            indices = []
            for i in zip(self.index.names, self.index[cadence]):
                if i[0] == "indices":
                    strlabel = f"{i[0]}: {i[1]}"
                elif ("cadence" in i[0]) or ("index" in i[0]):
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

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.to_array(), index=self.index, columns=self.columns, **self.user_kwargs
        )

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register(slice)
    @__getitem__.register(np.ndarray)
    @__getitem__.register(list)
    @__getitem__.register(range)
    def _(self, key):
        # Simple slice in time, results in DataCube
        return self.__class__.from_pandas(
            self.iloc[key],
            nrow=self.nrow,
            ncol=self.ncol,
            row_indices={name: None for name in self.row_names},
            col_indices={name: None for name in self.col_names},
            **self.user_kwargs,
        )

    @__getitem__.register
    def _(self, key: int):
        # Integer time, currently results in DataCube
        return self.__class__.from_pandas(
            self.iloc[np.atleast_1d(key)],
            nrow=self.nrow,
            ncol=self.ncol,
            **self.user_kwargs,
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
                **self.user_kwargs,
            )
        # If only two things passed
        if len(key) == 2:
            if isinstance(key[1], int):
                return DataSeries(self.iloc[time, key[1]], **self.user_kwargs)
            elif isinstance(key[1], slice):
                row = key[1]
                col = slice(self.ncol + 1)
            elif np.ndim(key[1]) == 2:
                aperture = key[1]
                # Passed an aperture with shape (nrow, ncol),
                # example:
                # aper = Cube.sum(axis=1) > 10000
                # Cube[:, aper]
                # needs to become frame of time-series
                return self[time].to_dataframe(*np.where(aperture), **self.user_kwargs)

            elif len(key[1]) == self.ncol * self.nrow:
                # Passed an aperture with shape(nrow*ncol)
                # example:
                # aper = Cube.row == 2
                # Cube[:, aper]
                nrow = len(self.columns.get_level_values(1)[key[1]].unique())
                ncol = len(self.columns.get_level_values(2)[key[1]].unique())
                return self.__class__.from_pandas(
                    self.iloc[time, key[1]],
                    nrow=nrow,
                    ncol=ncol,
                    **self.user_kwargs,
                )

        if len(key) == 3:
            row, col = key[1], key[2]

        # To be a a 3D dataset needs to pass slices or integers as row/column
        if isinstance(row, int) & isinstance(col, int):
            _, _, series = self._convert_to_series_index(row, col)
            return DataSeries(self.iloc[time, int(series[0])], **self.user_kwargs)
        elif (not isinstance(row, (slice))) | (not isinstance(col, (slice))):
            return self[time].to_dataframe(row, col, **self.user_kwargs)
        elif (isinstance(row, slice) & (row.step not in [None, 1])) | (
            isinstance(col, slice) & (col.step not in [None, 1])
        ):
            return self[time].to_dataframe(row, col, **self.user_kwargs)

        nrow, ncol, series_index = self._convert_to_series_index(row, col)
        return self.__class__.from_pandas(
            self.iloc[time, series_index],
            nrow=nrow,
            ncol=ncol,
            **self.user_kwargs,
        )

    def describe_cube(self, **printoptions):
        """Print a description of the Cube instance.

        This description prints information about the temporal and spatial
        indices available in the Cube. It also prints out any additional
        user-assigned properties given via the kwargs on initialization.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            max_name_len = max(map(len, self._metadata))
            print(repr(self) + " (ntime, nrow, ncol)")
            print()
            print("Time indices available: " + str(self.index.names))
            for key in self.index.names:
                print(
                    f"\t{key.ljust(max_name_len+1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print(f"Number of unique 'series': {len(self.series)}")
            print()
            print("Row names: " + str(self.row_names))
            for key in self.row_names:
                print(
                    f"\t{key.ljust(max_name_len+1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print("Column names: " + str(self.col_names))
            for key in self.col_names:
                print(
                    f"\t{key.ljust(max_name_len+1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print("User defined attributes accessible via `object.key`")
            print("(displaying only unique values)")
            for key in self._user_kwargs:
                print(
                    f'\t{key.ljust(max_name_len+1)}:\t{getattr(self, key, "Not defined")}'
                )

    @property
    def nseries(self):
        """Total number of time series contained in the cube"""
        return self.ncol * self.nrow

    @property
    def units(self):
        """Data units, if any"""
        return self._flux_units

    @units.setter
    def units(self, unit):
        # remove formatting, if present
        self._flux_units = str(unit)

    @property
    def styler(self):
        """The pandas.DataFrame styler for single cadence frames."""
        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, val: Styler):
        self._styler = val

    def stats_post_process(self, result, **kwargs):
        """Statistics post processer to format return data."""
        if kwargs.get("axis") in [0, "time"]:
            return result.to_numpy().reshape(self.nrow, self.ncol)
        elif kwargs.get("axis") in [1, "series"]:
            return self._series_class(result)
        else:
            return result

    def to_dataframe(
        self,
        row: Union[int, float, list, slice],
        col: Union[int, float, list, slice],
        **kwargs,
    ) -> Union[DataFrame, ErrorFrame]:
        """Convert Cube to Frame with the given row and column indices.

        Parameters
        ----------
        row: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of row indices to include.
        col: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of column indices to include.

        Returns
        -------
        Union[DataFrame, ErrorFrame]
            A Frame object of the same type as the input data, either
            DataFrame or ErrorFrame.
        """
        nrow, ncol, series_index = self._convert_to_series_index(row, col)
        return self._frame_class(
            self.iloc[:, series_index],
            index=self.index,
            columns=self.columns[series_index],
            nrow=nrow,
            ncol=ncol,
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
    def from_pandas(cls, data: pd.DataFrame, **kwargs):
        """Convert a pd.DataFrame to a DataCube

        Notes:
        This assumes no multi-indexing in the pandas dataframe.
        """
        return cls(data.to_numpy(), index=data.index, columns=data.columns, **kwargs)


class DataCube(
    Cube,
    StatsMixin,
):
    """A Cube object which contains data with time and 2 spatial dimensions."""

    _frame_class = DataFrame
    _series_class = DataSeries
    _pd_class = pd.DataFrame

    def __init__(self, *args, **kwargs):
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        super().__init__(*args, **kwargs)
        self._set_stats_methods()

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}"


class ErrorCube(
    Cube,
    ErrorStatsMixin,
):
    """A Cube object which contains errors with time and 2 spatial dimensions."""

    _frame_class = ErrorFrame
    _series_class = ErrorSeries
    _pd_class = pd.DataFrame

    def __init__(self, *args, **kwargs):
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        super().__init__(*args, **kwargs)
        self._set_errstats_methods()

    def __repr__(self):
        return f"📕 ErrorCube {self.ntime, self.nrow, self.ncol}"
