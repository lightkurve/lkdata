"""Classes and tools for working with 3 dimensional data."""

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
    MathMixin,
    StatsMixin,
    BoolMixin,
    BitwiseMixin,
)

log = logging.getLogger()


class Frame(
    ABC,
    MathMixin,
    AggMixin,
    ConvenienceMixins,
    pd.DataFrame,
):
    """Abstract dataclass for frame-like data with time and multiple series"""

    _pd_class = pd.DataFrame
    row_names = None
    col_names = None
    _user_kwargs = None
    _array = None

    def __init__(self, *args, **kwargs):
        index = kwargs.pop("index", None)
        time_indices = kwargs.pop("time_indices", None)
        index = self.parse_index(index, time_indices)
        if index.empty:
            index = None
        columns = kwargs.pop("columns", None)
        row_indices = kwargs.pop("row_indices", None)
        col_indices = kwargs.pop("col_indices", None)
        columns, kwargs["nrow"], kwargs["ncol"] = self.parse_columns(
            columns,
            row_indices,
            col_indices,
            kwargs.get("nrow", 0),
            kwargs.get("ncol", 0),
        )
        if columns.empty:
            columns = None
        for key, val in kwargs.items():
            if key not in (
                "ntime",
                "index",
                "columns",
                "time_indices",
                "row_indices",
                "col_indices",
            ):
                self._metadata.append(key)
                setattr(self, key, val)
                if key not in ("nrow", "ncol"):
                    self._user_kwargs.append(key)

        for key in self._metadata:
            kwargs.pop(key, None)

        pd.DataFrame.__init__(self, *args, index=index, columns=columns, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        self._array = self.to_numpy()
        self._include_convenience_index()
        self._include_convenience_columns()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.array, index=self.index, columns=self.columns, **self.user_kwargs
        )

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register(Iterable)
    @__getitem__.register(slice)
    def _(self, key):
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        return self.__class__.from_pandas(
            self.iloc[key],
            index=self.index[key],
            columns=self.columns,
            **init_kwds,
        )

    @__getitem__.register(int)
    def _(self, key):
        # bypassing pandas conversions to series when given int keys
        return self[key : key + 1]

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]

        if isinstance(key[1], slice):
            series_index = np.arange(self.nseries)[key[1]]
        elif isinstance(key[1], Iterable):
            series_index = key[1]
        else:
            return self._series_class(
                self.iloc[time_key, key[1]],
                index=self.index[time_key],
                **init_kwds,
            )

        return self.__class__.from_pandas(
            self.iloc[time_key, series_index],
            index=self.index[time_key],
            columns=self.columns[series_index],
            **init_kwds,
        )

    def _repr_html_(self):
        return repr(self) + super()._repr_html_()

    @property
    def array(self):
        return self._array

    def describe_frame(self, **printoptions):
        """Print a description of the Frame instance.

        This description prints information about the temporal and spatial
        indices available in the Frame. It also prints out any additional
        user-assigned properties given via the kwargs on initialization.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            max_name_len = max(map(len, self._metadata))
            print(repr(self) + " (ntime, nseries)")
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
            if hasattr(self, "row_names"):
                print("Row names: " + str(self.row_names))
                for key in self.row_names:
                    print(
                        f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                    )
                print()
            if hasattr(self, "col_names"):
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

    @property
    def nseries(self):
        """Number of series in the DataFrame"""
        return self.shape[1]

    @property
    def ntime(self):
        """Number of time frames"""
        return self.shape[0]

    def stats_post_process(self, result, **kwargs):
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


class DataFrame(StatsMixin, Frame):
    _series_class = DataSeries

    def __init__(self, *args, **kwargs):
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
        uncertainty = kwargs.pop("uncertainty", None)
        self._metadata = []
        self._user_kwargs = []
        super().__init__(*args, **kwargs)
        self.uncertainty = uncertainty
        self._set_stats_methods()

    def __repr__(self):
        return f"🟦 DataFrame {self.shape}"

    @staticmethod
    def from_pandas(data, **kwargs):
        """Convert a pd.DataFrame to a DataFrame"""
        return DataFrame(data, **kwargs)


class BoolFrame(
    BoolMixin,
    Frame,
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
        super().__init__(*args, dtype=bool, **kwargs)

    def __repr__(self):
        return f"⚫️⚪️ BoolFrame {self.shape}"


class BitwiseFrame(BitwiseMixin, Frame):
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
            display_as: str = "bitwise"
        """
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        data = BitwiseMixin._set_data_type_to_bitset(args[0])
        self._metadata = []
        self._user_kwargs = []
        kwargs["codes"] = kwargs.get("codes", {})
        self.codes = kwargs["codes"]
        values_display = kwargs.pop("values_display", None) or kwargs.pop(
            "display_as", "bitwise"
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
        if hasattr(self, "_styler"):
            return self._styler
        return None

    @styler.setter
    def styler(self, val: Styler):
        self._styler = val
