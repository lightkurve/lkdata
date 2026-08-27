"""Classes and tools for working with collections of Series products."""

import logging
from abc import ABC
from collections.abc import Iterable
from functools import singledispatchmethod

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from .dataseries import DataSeries, BoolSeries, BitwiseSeries
from .mixins import (
    AggMixin,
    ConvenienceMixins,
    StatsMixin,
    BoolMixin,
    BitwiseMixin,
)

log = logging.getLogger()


class SeriesCollection(
    ABC,
    AggMixin,
    ConvenienceMixins,
    pd.DataFrame,
):
    """Abstract dataclass for frame-like data with time and multiple series"""

    _pd_class = pd.DataFrame
    _nrow: int = 0
    _ncol: int = 0
    _row_names = None
    _col_names = None
    _user_kwargs = None

    def __init__(
        self, data, time_indices=None, row_indices=None, col_indices=None, **kwargs
    ):
        # Pandas DataFrame kwargs
        copy = kwargs.pop("copy", None)
        dtype = kwargs.pop("dtype", None)

        # Reserved names
        kwargs.pop("ntime", None)
        ntime = np.array(data).shape[0]
        nrow = kwargs.pop("nrow", 0) or 0
        ncol = kwargs.pop("ncol", 0) or 0
        index = kwargs.pop("index", None)
        if index is not None and not isinstance(index, pd.Index):
            index = pd.Index(index, name="given_index")
        columns = kwargs.pop("columns", None)
        nseries = np.array(data).shape[1]

        for key, val in kwargs.items():
            self._user_kwargs.append(key)  # for building new products
            self._metadata.append(key)  # for adding attrs to pandas DataFrame subclass
            setattr(self, key, val)

        index = self.parse_index(index, time_indices, ntime)

        columns, self._nrow, self._ncol = self.parse_columns(
            columns,
            row_indices,
            col_indices,
            nrow,
            ncol,
            nseries=nseries,
        )

        pd.DataFrame.__init__(
            self, data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.__post_init__()

    def __post_init__(self):
        self._include_convenience_index()
        self._include_convenience_columns()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.array, index=self.index, columns=self.columns, **self.user_kwargs
        )

    @singledispatchmethod
    def __getitem__(self, key):  # pragma: no cover
        return super().__getitem__(key)

    @__getitem__.register(Iterable)
    @__getitem__.register(slice)
    def _(self, key):
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        return self.__class__.from_pandas(
            self.iloc[key],
            **init_kwds,
        )

    @__getitem__.register(int)
    def _(self, key):
        # bypassing pandas conversions to series when given int keys
        return self[key : key + 1]

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        if not isinstance(time_key, slice):
            # return Frame or Series even if it's a single value
            time_key = np.atleast_1d(time_key)
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]

        if isinstance(key[1], slice):
            series_index = np.arange(self.nseries)[key[1]]
        elif isinstance(key[1], Iterable):
            series_index = key[1]
        elif isinstance(key[1], int):
            return self._series_class(
                self.iloc[time_key, key[1]],
                index=self.index[time_key],
                **init_kwds,
            )
        else:  # pragma: no cover
            raise ValueError(
                "Location based indexing can only have [integer, "
                "integer slice (START point is INCLUDED, END "
                "point is EXCLUDED), listlike of integers, "
                "boolean array] types"
            )

        return self.__class__.from_pandas(
            self.iloc[time_key, series_index],
            **init_kwds,
        )

    def _repr_html_(self):  # pragma: no cover
        return repr(self) + super()._repr_html_()

    @property
    def array(self):
        """Numpy array representation with shape (ntime, nseries)"""
        return self.to_numpy().reshape(self.ntime, self.nseries)

    def describe_collection(self, **printoptions):  # pragma: no cover
        """Print a description of the SeriesCollection instance.

        This description prints information about the temporal and spatial
        indices available in the SeriesCollection. It also prints out any additional
        user-assigned properties given via keyword arguments on initialization.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            max_name_len = max(map(len, self._metadata))
            print(repr(self) + " (ntime, nseries)")
            print()
            if (
                hasattr(self, "uncertainty")
                and issubclass(self.__class__, DataSeriesCollection)
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
            if hasattr(self, "row_names") and getattr(self, "row_names") is not None:
                print("Row names: " + str(self.row_names))
                for key in self.row_names:
                    print(
                        f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                    )
                print()
            if hasattr(self, "col_names") and getattr(self, "col_names") is not None:
                print("Column names: " + str(self.col_names))
                for key in self.col_names:
                    print(
                        f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                    )
                print()
            if self._user_kwargs is not None and len(self._user_kwargs) > 0:
                print("User defined attributes accessible via `object.key`")
                print("(displaying only unique values)")
                for key in self._user_kwargs:
                    print(
                        f"  {key.ljust(max_name_len + 1)} {type(getattr(self, key, None))}\t:\t{getattr(self, key, 'Not defined')}"
                    )

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, **kwargs):
        """Convert a pd.DataFrame to a SeriesCollection"""
        return cls(data.to_numpy(), index=data.index, columns=data.columns, **kwargs)

    @property
    def nseries(self):
        """Number of series in the SeriesCollection"""
        return len(self.columns)

    def _stats_post_process(self, result, **kwargs):
        axis = kwargs.pop("axis")
        uncertainty = kwargs.pop("uncertainty", None)
        if axis in [0, "time"]:
            if uncertainty:
                return result, uncertainty
            return result
        if axis in [1, "pixel"]:
            return self._series_class(result, uncertainty=uncertainty, **kwargs)
        else:
            return result


class DataSeriesCollection(StatsMixin, SeriesCollection):
    _series_class = DataSeries

    def __init__(
        self,
        data,
        uncertainty=None,
        time_indices=None,
        row_indices=None,
        col_indices=None,
        **kwargs,
    ):
        """
        Args:
            data: Union[List, np.ndarray],
            uncertainty: Union[List, np.ndarray] = None,
            index: pd.MultiIndex = None,
            columns: pd.MultiIndex = None,
            time_indices: Union[Dict, List, None] = None,
            row_indices: Union[Dict, List, None] = None,
            col_indices: Union[Dict, List, None] = None,
            dtype: type = float
        """
        self._metadata = []
        self._user_kwargs = []
        super().__init__(data, time_indices, row_indices, col_indices, **kwargs)
        self._array = self.to_numpy().reshape(self.ntime, self.nseries)
        self.uncertainty = uncertainty
        self._set_stats_methods()

    def __repr__(self):  # pragma: no cover
        return f"🟦 DataSeriesCollection {self.shape}, Uncertainty: {bool(self.uncertainty)}"


class BoolSeriesCollection(
    BoolMixin,
    SeriesCollection,
):
    """A Cube object which contains boolean values with time and 2 spatial dimensions."""

    _series_class = BoolSeries  # BoolSeries

    def __init__(self, *args, **kwargs):
        """
        Args:
            data: Union[List, np.ndarray],
            index: pd.MultiIndex = None,
            columns: pd.MultiIndex = None,
            time_indices: Union[Dict, List, None] = None,
            row_indices: Union[Dict, List, None] = None,
            col_indices: Union[Dict, List, None] = None,
        """
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata = []
        self._user_kwargs = []
        kwargs.pop("dtype", None)
        super().__init__(*args, dtype=bool, **kwargs)

    def __repr__(self):
        return f"⚫️⚪️ BoolFrame {self.shape}"


class BitwiseSeriesCollection(BitwiseMixin, SeriesCollection):
    """A Cube object which contains bitwise values with time and 2 spatial dimensions."""

    _series_class = BitwiseSeries

    def __init__(self, *args, **kwargs):
        """
        Args:
            data: Union[List, np.ndarray],
            index: pd.MultiIndex = None,
            columns: pd.MultiIndex = None,
            time_indices: Union[Dict, List, None] = None,
            row_indices: Union[Dict, List, None] = None,
            col_indices: Union[Dict, List, None] = None,
            display_as: str = "int"
        """
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        data = BitwiseMixin._set_data_type_to_bitset(args[0])
        self._metadata = []
        self._user_kwargs = []
        kwargs["codes"] = kwargs.get("codes", {})
        self.codes = kwargs["codes"]
        values_display = kwargs.pop("values_display", None) or kwargs.pop(
            "display_as", "int"
        )
        kwargs.pop("dtype", None)
        super().__init__(data, *args[1:], dtype=object, **kwargs)
        self.values_display = values_display  # set after dataframe init
        self._user_kwargs.append("values_display")

    def __repr__(self):
        return f"📗 BitwiseFrame {self.shape}"

    def _repr_html_(self):
        if hasattr(self, "_styler"):
            out0 = self.styler
        else:
            out0 = self.stylize_frame(self)
            self.styler = out0
        return f"""
        {repr(self)}
        {out0.to_html()}
        """

    @property
    def styler(self):
        """pd.Styler object associated with this BitwiseSeriesCollection.

        The Styler allows for customized display of the data in HTML format.

        Returns
        -------
        pd.Styler
            The associated Styler object, or None if not set.
        """

        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, val: Styler):
        self._styler = val
