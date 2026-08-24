"""Classes and tools for working with data cubes, continuous regions of time series data."""

import logging
from abc import ABC
from functools import singledispatchmethod
from typing import Union, List, Dict, Optional
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
from numpy.typing import ArrayLike

from .seriescollection import (
    DataSeriesCollection,
    BoolSeriesCollection,
    BitwiseSeriesCollection,
)
from .dataseries import DataSeries, BoolSeries, BitwiseSeries
from .mixins import (
    StatsMixin,
    BoolMixin,
    BitwiseMixin,
    AggMixin,
    ConvenienceMixins,
)

log = logging.getLogger()

# __all__ = ["DataCube", "BoolCube", "BitwiseCube"]


class Cube(
    ABC,
    AggMixin,
    ConvenienceMixins,
    pd.DataFrame,
):
    """
    A three-dimensional data structure representing time series of two-dimensional spatial data.

    This class extends pandas.DataFrame to handle 3D data with time and spatial dimensions.
    It provides methods for data manipulation, indexing, and statistical operations.

    Parameters
    ----------
    data : array-like
        The input data for the Cube. Should be 2D or 3D array-like.
    uncertainty : array-like
    time_indices : array-like or dictionary of array-like, optional
        Indices for the time dimension.
        If an array-like, the default name "time_index" will be assigned.
        If a dictionary, each entry will be added to the MultiIndex with the
        corresponding key. A default RangeIndex from 0 to the length of the
        given index (or data, when this argument is not provided) with the key
        "time_index" is added if the "time_index" key is not specified.
        If this argument and no keyword argument "index" or "ntime" are given,
        a RangeIndex from 0 to the size of the first dimension of the data is
        added as "time_index".
    row_indices : array-like or dictionary of array-like, optional
        Indices for the row dimension.
        If an array-like, the default name "row" will be assigned.
        If a dictionary, each entry will be added to the MultiIndex with the
        corresponding key.
        If this argument and "nrow" are not given, a default RangeIndex from 0
        to the size of the second dimension of the given data is added as "row".
    col_indices : array-like or dictionary of array-like, optional
        Indices for the column dimension.
        If an array-like, the default name "col" will be assigned.
        If a dictionary, each entry will be added to the MultiIndex with the
        corresponding key.
        If this argument and "ncol" are not given, a default RangeIndex from 0
        to the size of the third dimension of the given data is added as "col".
    **kwargs
        Any keyword arguments for constructing a pandas DataFrame, like `index` and
        `columns`, will be treated appropriately. Any unrecognized keys
        are stored as class attributes.
        If the given data is flattened `nrow` and `ncol` must be specified.
        The class will use data.reshape((ntime, nrow, ncol)) to store the data
        and generate automatic indices based on nrow and ncol, unless otherwise
        specified.


    Attributes
    ----------
    array
    nseries
    styler
    units
    values
    nrow : int, optional
        Number of rows in the spatial dimensions.
    ncol : int, optional
        Number of columns in the spatial dimensions.
    row_names : list of strings, optional
        Names of the row indices.
    col_names : list of strings, optional
        Names of the column indices.

    Methods
    -------
    describe_cube(**printoptions)
        Prints a description of the Cube instance.
    from_pandas(data: pd.DataFrame, **kwargs)
        Converts a pd.DataFrame to a Cube.
    make_cadence_label(cadence: int)
        Creates a formatted cadence label for the HTML representation.
    single_frame(cadence: int)
        Creates a stylized single cadence frame of the cube.
    to_seriescollection(row, col, **kwargs)
        Converts the Cube to a DataSeriesCollection with given row and column indices.
    """

    _pd_class = pd.DataFrame
    _nrow: int = 0
    _ncol: int = 0
    _row_names: Optional[List[str]] = None
    _col_names: Optional[List[str]] = None
    _user_kwargs: Optional[List[str]] = None

    def __init__(
        self,
        data: ArrayLike,
        time_indices: Optional[Union[Dict, List]] = None,
        row_indices: Optional[Union[Dict, List]] = None,
        col_indices: Optional[Union[Dict, List]] = None,
        **kwargs,
    ):
        # Pandas DataFrame kwargs
        copy = kwargs.pop("copy", None)
        dtype = kwargs.pop("dtype", None)

        # Reserved names
        kwargs.pop("ntime", None)
        ntime = np.array(data).shape[0]
        self._nrow = kwargs.pop("nrow", None)
        self._ncol = kwargs.pop("ncol", None)
        columns = kwargs.pop("columns", None)
        index = kwargs.pop("index", None)
        if index is not None and not isinstance(index, pd.Index):
            index = pd.Index(index, name="given_index")

        # User defined properties, stored as custom attributes
        for key, val in kwargs.items():
            self._user_kwargs.append(key)  # for building new products
            self._metadata.append(key)  # for adding attrs to pandas DataFrame subclass
            setattr(self, key, val)
        data = self._preprocess_data(data)
        index = self.parse_index(index, time_indices, ntime)
        columns, self._nrow, self._ncol = self.parse_columns(
            columns, row_indices, col_indices, self.nrow, self.ncol, continuous=True
        )

        if len(data) != len(index):
            raise ValueError("Length of index does not match shape of data.")
        if len(columns) != data.shape[1]:
            raise ValueError("Number of columns does not match shape of data.")
        super().__init__(data, index=index, columns=columns, dtype=dtype, copy=copy)
        self._include_convenience_index()
        self._include_convenience_columns()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.array, index=self.index, columns=self.columns, **self.user_kwargs
        )

    @singledispatchmethod
    def __getitem__(self, key):  # pragma: no cover
        """
        Note: keys given to __getitem__ are interpreted as iloc indices.
        """
        raise KeyError("Unsupported type given for key.")

    @__getitem__.register(int)
    @__getitem__.register(slice)
    @__getitem__.register(np.ndarray)
    @__getitem__.register(list)
    @__getitem__.register(range)
    def _(self, key):
        """Simple slice only on time, results in Cube"""
        if isinstance(key, int):
            key = [key]
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        return self.__class__.from_pandas(
            self.iloc[key],
            nrow=self.nrow,
            ncol=self.ncol,
            **init_kwds,
        )

    @__getitem__.register(tuple)
    def _(self, key):
        """Slice on multiple axes."""
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
            return self[time].to_seriescollection(row, col, **init_kwds)
        elif (isinstance(row, slice) & (row.step not in [None, 1])) | (
            isinstance(col, slice) & (col.step not in [None, 1])
        ):
            return self[time].to_seriescollection(row, col, **init_kwds)

        nrow, ncol, series_index = self._convert_to_series_index(row, col)

        return self.__class__.from_pandas(
            self.iloc[time, series_index],
            nrow=nrow,
            ncol=ncol,
            **init_kwds,
        )

    def __repr__(self):
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            return f"Cube + Uncertainty {self.ntime, self.nrow, self.ncol}"
        return f"Cube {self.ntime, self.nrow, self.ncol}"

    def __str__(self):
        return self.__repr__()

    def _convert_to_series_index(self, row, col):
        """Convert (row, col) input to DataFrame series indices

        Where a Cube is subscripted with row and col inputs, either slices,
        lists of integers, a single integer, or a mixture thereof, this method
        converts such inputs into positional column indices and also returns
        the new dimensions for nrow and ncol.

        Parameters:
            row : int, slice, or array-like of int
            col : int, slice, or array-like of int

        Returns:
            nrow: int
            ncol: int
            series_index: ndarray
        """
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
        log.info("data.ndim = %s, data.shape= %s", data.ndim, data.shape)
        if data.ndim == 2:
            if (self.nrow is None) | (self.ncol is None):
                raise ValueError(
                    """
                Must set `nrow` and `ncol` when giving data as a 2D array.
                """
                )
        elif data.ndim == 3:
            # Reshape for a 2D Pandas DataFrame
            self._nrow = data.shape[1]
            self._ncol = data.shape[2]
            # self._set_dim("nrow", data.shape[1])
            # self._set_dim("ncol", data.shape[2])
            data = np.hstack(data.transpose([1, 0, 2]))
        else:
            raise ValueError("""Dimension of given data not interpretable as a Cube""")
        return data

    def _repr_html_(self):
        if self.ntime == 0:
            return repr(self)

        if hasattr(self, "_styler"):
            out0 = self.styler
        else:
            df = self.get_single_frame(0)
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

    def _stats_post_process(self, result, **kwargs):
        """Statistics post processer to format return data."""
        axis = kwargs.pop("axis")
        uncertainty = kwargs.pop("uncertainty", None)
        if axis is None:
            if uncertainty:
                return result, uncertainty
            else:
                return result
        elif axis in [0, "time"]:
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
    def array(self):
        """Numpy array representation with shape (ntime, nrow, ncol)"""

        return self.to_numpy().reshape(self.ntime, self.nrow, self.ncol)

    @property
    def col_names(self) -> List[str]:
        """List of distinct column names

        Note: this is distinct from pandas.Columns. The columns referred
        to here correspond to the columnal spatial portion of the given data
        rather than the tabular columns of the DataFrame.
        """
        if self._col_names is None:
            self._col_names = []
        return self._col_names

    @property
    def row_names(self) -> List[str]:
        """List of distinct row names

        Note:  The rows referred
        to here correspond to the row spatial portion of the given data
        rather than the tabular rows of the DataFrame (which correpsond
        to time indices).
        """
        if self._row_names is None:
            self._row_names = []
        return self._row_names

    @property
    def ncol(self):
        """Number of distinct columns in the data

        Note: this is distinct from  the number of pandas.Columns.
        The columns referred to here correspond to the columnal spatial portion
        of the given data rather than the tabular columns of the DataFrame.
        """
        return self._ncol

    @property
    def nrow(self):
        return self._nrow

    @property
    def nseries(self):
        """Total number of time series contained in the cube"""
        return self.ncol * self.nrow

    @property
    def styler(self):
        """The pandas.DataFrame styler for single cadence frames."""
        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, val: Styler):
        self._styler = val

    @property
    def units(self):
        """Data units, if any"""
        return self._flux_units

    @units.setter
    def units(self, unit):
        # remove formatting, if present
        self._flux_units = str(unit)

    def describe_cube(self, **printoptions):  # pragma: no cover
        """Print a description of the Cube instance.

        This description prints information about the temporal and spatial
        indices available in the Cube. It also prints out any additional
        user-assigned properties given via keyword arguments on initialization.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            max_name_len = max(map(len, self._metadata))
            print(repr(self) + " (ntime, nrow, ncol)")
            print(f"pd.DataFrame shape: {self.shape}")
            print()
            if (
                hasattr(self, "uncertainty")
                and issubclass(self.__class__, DataCube)
                and bool(self.uncertainty)
            ):
                print("Uncertainty:")
                print(
                    f"  uncertainty\t:\t{type(self.uncertainty).__name__}(np.ndarray{self.uncertainty.shape})"
                )

            print()
            print("Time indices available: " + str(self.index.names))
            for key in self.index.names:
                print(
                    f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print(f"Number of unique 'series': {len(self.series)}")
            print()
            print("Row names: " + str(self._row_names))
            for key in self._row_names:
                print(
                    f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            print()
            print("Column names: " + str(self._col_names))
            for key in self._col_names:
                print(
                    f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            if len(self._user_kwargs) == 0:
                return
            print()
            print("User defined attributes accessible via `object.key`")
            print("(displaying only unique values)")
            for key in self._user_kwargs:
                print(
                    f"  {key.ljust(max_name_len + 1)} {type(getattr(self, key, None))}\t:\t{getattr(self, key, 'Not defined')}"
                )

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        row_names: Optional[Union[str, list]] = None,
        col_names: Optional[Union[str, list]] = None,
        nrow: Optional[int] = None,
        ncol: Optional[int] = None,
        **kwargs,
    ):
        """Convert a pd.DataFrame to a DataCube

        Parameters
        ----------
        data : pandas DataFrame
        row_name : str or list of strings, optional
            Name of "row" index in DataFrame.columns if columns is a MultiIndex.
        col_name : str or list of strings, optional
            Name of "col" index in Dataframe.columns if columns is a MultiIndex.
        nrow : int, optional
            number of rows to be inferred from the DataFrame.columns. Ignored
            if row_name is given.
        ncol : int, optional
            number of columns to be inferred from the DataFrame.columns. Ignored
            if col_name is given.

        Note:
            Keywords `index` and `columns` may not  be specified, they are inferred
            from the pandas DataFrame.

        """
        if row_names:
            if isinstance(row_names, str):
                row_indices = {
                    row_names: data.columns.get_level_values(row_names).values
                }
            elif isinstance(row_names, list):
                row_indices = {
                    n: data.columns.get_level_values(n).values for n in row_names
                }
            else:
                raise ValueError(
                    f"`row_names` must be int or list, got {type(row_names)}"
                )
            kwargs["row_indices"] = row_indices
            kwargs["nrow"] = len(np.unique(list(row_indices.values())[0]))
        elif nrow:
            kwargs["nrow"] = nrow
        else:
            raise KeyError("One of `row_names` or `nrow` must be specified.")

        if col_names:
            if isinstance(col_names, str):
                col_indices = {
                    col_names: data.columns.get_level_values(col_names).values
                }
            elif isinstance(col_names, list):
                col_indices = {
                    n: data.columns.get_level_values(n).values for n in col_names
                }
            else:
                raise ValueError(
                    f"`col_names` must be int or list, got {type(col_names)}"
                )
            kwargs["col_indices"] = col_indices
            kwargs["ncol"] = len(np.unique(list(col_indices.values())[0]))
        elif ncol:
            kwargs["ncol"] = ncol
        else:
            raise KeyError("One of `col_names` or `ncol` must be specified.")

        return cls(data.to_numpy(), index=data.index, columns=data.columns, **kwargs)

    def make_cadence_label(self, cadence: int):
        """Create a formatted cadence label for the HTML repr

        Parameters
        ----------
        cadence : int
            Cadence for which a label should be created

        Returns
        -------
        str
            Cadence label for the HTML repr
        """
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

    def get_single_frame(self, cadence: int) -> pd.DataFrame:
        """Create a stylized single cadence frame of a datacube

        This is distinct from to_seriescollection() and from retreiving a single
        cadence of a DataCube. The former returns a pandas DataFrame with all
        pixels along the column axis and all cadences along the index axis.
        The latter returns a DataCube with a single cadence, time information
        is retained and remains the first index.

        This returns a pandas DataFrame with rows on the index axis, and
        columns along the column axis. Time information is lost.
        """
        cadence = int(np.floor(cadence))

        row = getattr(self, self._row_names[0])
        col = getattr(self, self._col_names[0])
        df = pd.DataFrame(
            self.array[cadence],
            index=pd.Series(row[:: self.ncol], name=self._row_names[0]),
            columns=pd.MultiIndex.from_product(
                [[self._col_names[0]], pd.Series(col[: self.ncol])]
            ),
        )
        return df

    def stylize_frame(self, df, **kwargs):
        """Stylize a pandas.DataFrame for display.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be stylized.
        **kwargs : dict
            Additional keyword arguments for styling.

        Returns
        -------
        pandas.io.formats.style.Styler
            The stylized DataFrame.

        Notes
        -----
        This method applies various styling options to the DataFrame,
        including background gradient, precision formatting, and table styles.
        """
        out = Styler(df)
        if "label" in kwargs:
            out = out.set_caption(kwargs.pop("label"))
        if self._stats_type == "error" or all(out.data.dtypes == float):
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

    def to_seriescollection(
        self,
        row: Union[int, float, list, slice],
        col: Union[int, float, list, slice],
        **kwargs,
    ):
        """Convert lkdata.Cube to lkdata.SeriesCollection with the given row and column.

        Parameters
        ----------
        row: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of row indices to include.
        col: Union[int, float, List[Union[int, float]], slice]
            Index/list of indices or slice of column indices to include.

        Returns
        -------
        SeriesCollection
            A SeriesCollection object of the same type as the input data.
        """
        nrow, ncol, series_index = self._convert_to_series_index(row, col)

        return self._collection_class(
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
    StatsMixin,
    Cube,
):
    """A Cube object which contains data with time and 2 spatial dimensions.


    Parameters
    ----------
    data : ArrayLike
        The input data for the Cube. Should be 2D or 3D array-like.
    uncertainty : Union[List, ArrayLike]
    time_indices : Union[Dict, List, None], optional
        Indices for the time dimension.
    row_indices : Union[Dict, List, None], optional
        Indices for the row dimension.
    col_indices : Union[Dict, List, None], optional
        Indices for the column dimension.
    **kwargs
        Any keyword arguments for constructing a pandas DataFrame, like `index` and
        `columns`, will be treated appropriately. Any unrecognized keys
        are stored as class attributes.


    Attributes
    ----------
    array
    nseries
    styler
    units
    values
    nrow : int, optional
        Number of rows in the spatial dimensions.
    ncol : int, optional
        Number of columns in the spatial dimensions.
    row_names : list of strings, optional
        Names of the row indices.
    col_names : list of strings, optional
        Names of the column indices.

    Methods
    -------
    describe_cube(**printoptions)
        Prints a description of the Cube instance.
    from_pandas(data: pd.DataFrame, **kwargs)
        Converts a pd.DataFrame to a Cube.
    make_cadence_label(cadence: int)
        Creates a formatted cadence label for the HTML representation.
    single_frame(cadence: int)
        Creates a stylized single cadence frame of the cube.
    to_seriescollection(row, col, **kwargs)
        Converts the Cube to a DataSeriesCollection with given row and column indices.

    """

    _collection_class = DataSeriesCollection
    _series_class = DataSeries

    def __init__(
        self,
        data: ArrayLike,
        uncertainty: Optional[Union[List, np.ndarray]] = None,
        time_indices: Optional[Union[Dict, List, None]] = None,
        row_indices: Optional[Union[Dict, List]] = None,
        col_indices: Optional[Union[Dict, List]] = None,
        **kwargs,
    ):
        self.units = getattr(data, "unit", kwargs.pop("units", ""))
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
        self._array = self.to_numpy().reshape(self.ntime, self.nrow, self.ncol)
        self.uncertainty = uncertainty
        if self.uncertainty.array is not None:
            self.uncertainty = uncertainty.reshape(self.array.shape)
        self._set_stats_methods()

    def __repr__(self):
        return f"📘 DataCube {self.ntime, self.nrow, self.ncol}, Uncertainty: {bool(self.uncertainty)}"


class BoolCube(
    BoolMixin,
    Cube,
):
    """A Cube object which contains boolean entries.


    Parameters
    ----------
    data : ArrayLike[bool]
        The input data for the Cube. Should be 2D or 3D array-like.
    time_indices : Union[Dict, List], optional
        Indices for the time dimension.
    row_indices : Union[Dict, List], optional
        Indices for the row dimension.
    col_indices : Union[Dict, List], optional
        Indices for the column dimension.
    **kwargs
        Any keyword arguments for constructing a pandas DataFrame, like `index` and
        `columns`, will be treated appropriately. Any unrecognized keys
        are stored as class attributes.


    Attributes
    ----------
    array
    nseries
    styler
    units
    values
    nrow : int, optional
        Number of rows in the spatial dimensions.
    ncol : int, optional
        Number of columns in the spatial dimensions.
    row_names : list of strings, optional
        Names of the row indices.
    col_names : list of strings, optional
        Names of the column indices.

    Methods
    -------
    describe_cube(**printoptions)
        Prints a description of the Cube instance.
    from_pandas(data: pd.DataFrame, **kwargs)
        Converts a pd.DataFrame to a Cube.
    make_cadence_label(cadence: int)
        Creates a formatted cadence label for the HTML representation.
    single_frame(cadence: int)
        Creates a stylized single cadence frame of the cube.
    to_seriescollection(row, col, **kwargs)
        Converts the Cube to a DataSeriesCollection with given row and column indices.

    """

    _collection_class = BoolSeriesCollection
    _series_class = BoolSeries

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
    """A Cube object which contains bitwise entries.


    Parameters
    ----------
    data : ArrayLike
        The input data for the Cube. Should be 2D or 3D array-like.
    time_indices : Union[Dict, List], optional
        Indices for the time dimension.
    row_indices : Union[Dict, List], optional
        Indices for the row dimension.
    col_indices : Union[Dict, List], optional
        Indices for the column dimension.
    code_dict : Dict, optional
        A dictionary mapping bit values to their definitions.
    display_as : str, optional
        How to display the values. Options are "int", "bitset",
        or "detailed".
    **kwargs
        Any keyword arguments for constructing a pandas DataFrame, like `index` and
        `columns`, will be treated appropriately. Any unrecognized keys
        are stored as class attributes.


    Attributes
    ----------
    array
    nseries
    styler
    units
    values
    nrow : int, optional
        Number of rows in the spatial dimensions.
    ncol : int, optional
        Number of columns in the spatial dimensions.
    row_names : list of strings, optional
        Names of the row indices.
    col_names : list of strings, optional
        Names of the column indices.

    Methods
    -------
    describe_cube(**printoptions)
        Prints a description of the Cube instance.
    from_pandas(data: pd.DataFrame, **kwargs)
        Converts a pd.DataFrame to a Cube.
    make_cadence_label(cadence: int)
        Creates a formatted cadence label for the HTML representation.
    single_frame(cadence: int)
        Creates a stylized single cadence frame of the cube.
    to_seriescollection(row, col, **kwargs)
        Converts the Cube to a DataSeriesCollection with given row and column indices.

    """

    _collection_class = BitwiseSeriesCollection
    _series_class = BitwiseSeries

    def __init__(
        self,
        data: Union[List, np.ndarray],
        time_indices: Optional[Union[Dict, List]] = None,
        row_indices: Optional[Union[Dict, List]] = None,
        col_indices: Optional[Union[Dict, List]] = None,
        code_dict: Optional[Dict] = None,
        display_as: str = "int",
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
        data = BitwiseMixin._set_data_type_to_bitset(data)
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        kwargs.pop("dtype", None)
        if code_dict is None:
            code_dict = {}
        self.codes = code_dict
        display_as = kwargs.pop("values_display", None) or display_as
        super().__init__(
            data, time_indices, row_indices, col_indices, dtype=object, **kwargs
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
        allowed = {"int", "bitset", "detailed"}
        if value.lower() not in allowed:
            raise AttributeError(f"Display must be one of {allowed}.")
        self._values_display = value.lower()
        df = self.get_single_frame(0)
        label = self.make_cadence_label(0)
        self.styler = self.stylize_frame(df, label=label, cmap="gray")
