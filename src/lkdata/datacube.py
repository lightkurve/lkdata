"""Classes and tools for working with 3 dimensional data."""

import logging
from abc import ABC
from functools import singledispatchmethod
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from typing import Union, List, Dict, Optional

from .dataframe import DataFrame, BoolFrame, BitwiseFrame
from .dataseries import DataSeries, BoolSeries, BitwiseSeries
from .mixins import (
    StatsMixin,
    MathMixin,
    BoolMixin,
    BitwiseMixin,
    AggMixin,
    ConvenienceMixins,
)

log = logging.getLogger()


class Cube(
    ABC,
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
    _array = None

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
        dtype = kwargs.pop("dtype", float)

        for key, val in kwargs.items():
            if key not in ("ntime", "nrow", "ncol", "index", "columns"):
                self._user_kwargs.append(key)
                self._metadata.append(key)
                setattr(self, key, val)

        data = self._preprocess_data(data)
        index = self.parse_index(index, time_indices, self.ntime)
        columns, self.nrow, self.ncol = self.parse_columns(
            columns, row_indices, col_indices, self.nrow, self.ncol, continuous=True
        )

        super().__init__(data, index=index, columns=columns, dtype=dtype)
        self._array = self.to_numpy().reshape(self.ntime, self.nrow, self.ncol)
        self._include_convenience_index()
        self._include_convenience_columns()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.array, index=self.index, columns=self.columns, **self.user_kwargs
        )

    @singledispatchmethod
    def __getitem__(self, key):
        """
        Note: keys given to __getitem__ are interpreted as iloc indices.
        """
        pass

    @__getitem__.register(slice)
    @__getitem__.register(np.ndarray)
    @__getitem__.register(list)
    @__getitem__.register(range)
    def _(self, key):
        # Simple slice in time, results in DataCube
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        return self.__class__.from_pandas(
            self.iloc[key],
            nrow=self.nrow,
            ncol=self.ncol,
            **init_kwds,
        )

    @__getitem__.register
    def _(self, key: int):
        # Integer time, currently results in DataCube
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        return self.__class__.from_pandas(
            self.iloc[np.atleast_1d(key)],
            nrow=self.nrow,
            ncol=self.ncol,
            **init_kwds,
        )

    @__getitem__.register
    def _(self, key: tuple):
        time = key[0]
        init_kwds = self.user_kwargs.copy()
        if len(key) == 1:
            return self[key[0]]

        if isinstance(key[0], (int, list, np.ndarray)):
            time = np.atleast_1d(time)
        elif isinstance(key[0], slice):
            time = range(self.ntime)[time]
        else:
            raise ValueError(f"Can not parse time {key[0]}")

        # If only two things passed
        if len(key) == 2:
            if isinstance(key[1], int):
                if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
                    init_kwds["uncertainty"] = self.uncertainty[key[0], key[1]]
                return DataSeries(self.iloc[time, key[1]], **init_kwds)
            elif isinstance(key[1], slice):
                row = key[1]
                col = slice(self.ncol + 1)
            elif np.ndim(key[1]) == 2:
                # Passed an aperture with shape (nrow, ncol),
                # example:
                # aper = Cube.sum(axis=1) > 10000
                # Cube[:, aper]
                # needs to become frame of time-series
                row, col = np.where(key[1])
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
                    **init_kwds,
                )

        elif len(key) == 3:
            row, col = key[1], key[2]
        else:
            raise KeyError("Too many values passed for key.")

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key[0], row, col]

        # To be a a 3D dataset needs to pass slices or integers as row/column
        if isinstance(row, int) & isinstance(col, int):
            _, _, series = self._convert_to_series_index(row, col)
            return self._series_class.from_pandas(
                self.iloc[time, int(series[0])], **init_kwds
            )
        elif (not isinstance(row, (slice))) | (not isinstance(col, (slice))):
            return self[time].to_dataframe(row, col, **init_kwds)
        elif (isinstance(row, slice) & (row.step not in [None, 1])) | (
            isinstance(col, slice) & (col.step not in [None, 1])
        ):
            return self[time].to_dataframe(row, col, **init_kwds)

        nrow, ncol, series_index = self._convert_to_series_index(row, col)

        return self.__class__.from_pandas(
            self.iloc[time, series_index],
            nrow=nrow,
            ncol=ncol,
            **init_kwds,
        )

    def __repr__(self):
        return f"Cube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

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

    def _repr_html_(self):
        if hasattr(self, "_styler"):
            out0 = self.styler
        else:
            df = self.single_frame(0)
            label = self.make_cadence_label(0)
            out0 = self.stylize_frame(df, label=label, cmap="gray")
            self.styler = out0

        if self.shape[0] > 1:
            hidden_frames = f"[+{self.shape[0] - 1} cadences]"
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

    def _set_dim(self, attr, val):
        if getattr(self, attr, None) is None:
            setattr(self, attr, val)
        elif getattr(self, attr) != val:
            raise ValueError(f"Given {attr} does not match given data shape {val}")

    @property
    def array(self):
        return self._array

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
            print(f"pd.DataFrame shape: {self.shape}")
            print()
            if hasattr(self, "uncertainty"):
                print("Uncertainty:")
                try:
                    print(
                        f"\tuncertainty\t:\tUncertainty(np.ndarray{self.uncertainty.shape})"
                    )
                except AttributeError:
                    print("\tuncertainty\t:\tUncertainty(None)")
            print()
            print("Time indices available: " + str(self.index.names))
            for key in self.index.names:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print(f"Number of unique 'series': {len(self.series)}")
            print()
            print("Row names: " + str(self.row_names))
            for key in self.row_names:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print("Column names: " + str(self.col_names))
            for key in self.col_names:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print("User defined attributes accessible via `object.key`")
            print("(displaying only unique values)")
            for key in self._user_kwargs:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not defined')}"
                )

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, **kwargs):
        """Convert a pd.DataFrame to a DataCube

        Notes:
        This assumes no multi-indexing in the pandas dataframe.
        """
        return cls(data.to_numpy(), index=data.index, columns=data.columns, **kwargs)

    def make_cadence_label(self, cadence: int):
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
        label = "<br>" + "<br>".join(indices)
        return label

    @property
    def nseries(self):
        """Total number of time series contained in the cube"""
        return self.ncol * self.nrow

    def single_frame(self, cadence: int) -> pd.DataFrame:
        """Create a stylized single cadence frame of a datacube

        This is distinct from to_dataframe() and from retreiving a single
        cadence of a DataCube. The former returns a pandas DataFrame with all
        pixels along the column axis and all cadences along the index axis.
        The latter returns a DataCube with a single cadence, time information
        is retained and remains the first index.

        This returns a pandas DataFrame with rows on the index axis, and
        columns along the column axis. Time information is lost.
        """
        cadence = int(np.floor(cadence))

        row = getattr(self, self.columns.names[1])
        col = getattr(self, self.columns.names[2])
        df = pd.DataFrame(
            self.array[cadence],
            index=pd.Series(row[:: self.ncol], name=self.columns.names[1]),
            columns=pd.MultiIndex.from_product(
                [[self.columns.names[2]], pd.Series(col[: self.ncol])]
            ),
        )
        return df

    def stats_post_process(self, result, **kwargs):
        """Statistics post processer to format return data."""
        axis = kwargs.pop("axis")
        uncertainty = kwargs.pop("uncertainty", None)
        if axis in [0, "time"]:
            if uncertainty:
                return (
                    result.reshape(self.nrow, self.ncol),
                    uncertainty.array.reshape(self.nrow, self.ncol),
                )
            else:
                return result.reshape(self.nrow, self.ncol)
        elif axis in [1, "series"]:
            index = kwargs.get("index", None)
            return self._series_class(result, uncertainty=uncertainty, index=index)
        else:
            return result

    @property
    def styler(self):
        """The pandas.DataFrame styler for single cadence frames."""
        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, val: Styler):
        self._styler = val

    def stylize_frame(self, df, **kwargs):
        out = Styler(df)
        if "label" in kwargs:
            out = out.set_caption(kwargs.pop("label"))
        if self._stats_type == "error":
            out = out.format(precision=3)
        else:
            out = out.format(precision=0, thousands=",")

        vmin = kwargs.pop("vmin", df.min(axis=None))
        vmax = kwargs.pop("vmax", df.max(axis=None))

        out = out.background_gradient(axis=None, vmin=vmin, vmax=vmax, **kwargs)

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

    def to_dataframe(
        self,
        row: Union[int, float, list, slice],
        col: Union[int, float, list, slice],
        **kwargs,
    ) -> DataFrame:
        """Convert lkdata.Cube to lkdata.Frame with the given row and column.

        Parameters
        ----------
        row: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of row indices to include.
        col: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of column indices to include.

        Returns
        -------
        Frame
            A Frame object of the same type as the input data.
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

    @property
    def units(self):
        """Data units, if any"""
        return self._flux_units

    @units.setter
    def units(self, unit):
        # remove formatting, if present
        self._flux_units = str(unit)

    @property
    def values(self):
        return super().values


class DataCube(
    MathMixin,
    StatsMixin,
    Cube,
):
    """A Cube object which contains data with time and 2 spatial dimensions."""

    _frame_class = DataFrame
    _series_class = DataSeries
    _pd_class = pd.DataFrame

    def __init__(
        self,
        data: Union[List, np.ndarray],
        uncertainty: Union[List, np.ndarray] = None,
        time_indices: Union[Dict, List, None] = None,
        row_indices: Union[Dict, List, None] = None,
        col_indices: Union[Dict, List, None] = None,
        **kwargs,
    ):
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata: List[str] = ["uncertainty"]
        self._user_kwargs: List[str] = []

        super().__init__(
            data=data,
            time_indices=time_indices,
            row_indices=row_indices,
            col_indices=col_indices,
            **kwargs,
        )

        self.uncertainty = uncertainty
        if self.uncertainty.array is not None:
            self.uncertainty = uncertainty.reshape(self.array.shape)
        self._set_stats_methods()

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}"


class BoolCube(
    BoolMixin,
    Cube,
):
    """A Cube object which contains boolean values with time and 2 spatial dimensions."""

    _frame_class = BoolFrame
    _series_class = BoolSeries
    _pd_class = pd.DataFrame

    def __init__(
        self,
        data: Union[List, np.ndarray],
        time_indices: Union[Dict, List, None] = None,
        row_indices: Union[Dict, List, None] = None,
        col_indices: Union[Dict, List, None] = None,
        **kwargs,
    ):
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        super().__init__(
            data, time_indices, row_indices, col_indices, dtype=bool, **kwargs
        )

    def __repr__(self):
        return f"⚫️⚪️ BoolCube {self.ntime, self.nrow, self.ncol}"


class BitwiseCube(BitwiseMixin, Cube):
    """A Cube object which contains bitwise values with time and 2 spatial dimensions."""

    _frame_class = BitwiseFrame
    _series_class = BitwiseSeries
    _pd_class = pd.DataFrame

    def __init__(
        self,
        data: Union[List, np.ndarray],
        time_indices: Union[Dict, List, None] = None,
        row_indices: Union[Dict, List, None] = None,
        col_indices: Union[Dict, List, None] = None,
        code_dict: Dict = None,
        display_as: str = "bitwise",
        **kwargs,
    ):
        """A Cube object which contains bitwise values.

        Parameters
        ----------
            data : Union[List, np.ndarray]
                The input data for the BitwiseCube. Values must be integers or
                sets of integers, or bitwise strings.
            time_indices : Union[Dict, List, None], optional
                Indices for the time dimension.
            row_indices : Union[Dict, List, None], optional
                Indices for the row dimension.
            col_indices : Union[Dict, List, None], optional
                Indices for the column dimension.
            code_dict : Dict, optional
                A dictionary mapping bit values to their definitions.
            display_as : str, optional
                How to display the values. Options are "bitwise", "parsed",
                or "detailed".
            **kwargs
                Additional keyword arguments to pass to the parent class.

        Attributes
        ----------
            codes : Dict
                A dictionary mapping bit values to their meanings.
            values_display : str
                The current display mode for values.
        """
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        data = BitwiseMixin._set_data_type_to_int(data)
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        if code_dict is None:
            code_dict = {}
        self.codes = code_dict
        super().__init__(
            data, time_indices, row_indices, col_indices, dtype=int, **kwargs
        )
        self.values_display = display_as
        self._user_kwargs.append("values_display")

    def __repr__(self):
        return f"📗 BitwiseCube {self.ntime, self.nrow, self.ncol}"

    @property
    def values_display(self):
        return self._values_display

    @values_display.setter
    def values_display(self, value):
        allowed = {"bitwise", "parsed", "detailed"}
        if value.lower() not in allowed:
            raise AttributeError(f"Display must be one of {allowed}.")
        self._values_display = value.lower()
        df = self.single_frame(0)
        label = self.make_cadence_label(0)
        self.styler = self.stylize_frame(df, label=label, cmap="gray")
