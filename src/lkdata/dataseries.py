"""Classes and tools for working with 1 dimensional data."""

import logging
from abc import ABC
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .mixins import (
    AggMixin,
    ConvenienceMixins,
    MathMixin,
    StatsMixin,
    BoolMixin,
    BitwiseMixin,
)

log = logging.getLogger()


class Series(
    ABC,
    AggMixin,
    ConvenienceMixins,
    pd.Series,
):
    """Abstract pd.Series-like dataclass with additional methods."""

    _pd_class = pd.Series
    _user_kwargs: Optional[List[str]] = None

    def __init__(
        self,
        data: Union[List, np.ndarray, Dict],
        time_indices: Union[Dict, List, None] = None,
        **kwargs,
    ):
        time_indices = time_indices or {}

        # Pandas Series kwargs
        kwargs.pop("ntime", None)
        copy = kwargs.pop("copy", None)
        dtype = kwargs.pop("dtype", None)
        name = kwargs.pop("name", None)
        pdseries = pd.Series(data)

        index = kwargs.pop("index", None)
        index = self.parse_index(index, time_indices, pdseries.shape[0])

        # User defined properties, stored as custom attributes
        for key, val in kwargs.items():
            self._user_kwargs.append(key)
            self._metadata.append(key)
            setattr(self, key, val)

        super().__init__(data, index=index, dtype=dtype, name=name, copy=copy)
        self.__post_init__()

    def __post_init__(self):
        self._array = self.to_numpy()
        self._include_convenience_index()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(self.array, index=self.index, **self.user_kwargs)

    def __getitem__(self, key):
        result_data = super().__getitem__(key)
        result_index = self.index[key]
        init_kwds = self.user_kwargs.copy()
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            init_kwds["uncertainty"] = self.uncertainty[key]
        if isinstance(key, int):
            if "uncertainty" in init_kwds:
                return result_data, init_kwds["uncertainty"]
            return result_data
        else:
            return self.__class__(result_data, index=result_index, **init_kwds)

    @property
    def array(self):
        """Numpy array representation"""
        return self.to_numpy()

    def describe_series(self, **printoptions):  # pragma: no cover
        """Print a description of the Series instance.

        This description prints information about the temporal indices
        available in the Series. It also prints out any additional
        user-assigned properties given via keyword arguments on initialization.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            max_name_len = max(map(len, self._metadata))
            print(f"{repr(self)} {self.shape} (ntime)")
            print()
            if hasattr(self, "uncertainty") and issubclass(self.__class__, DataSeries):
                print("Uncertainty:")
                try:
                    print(
                        f"\tuncertainty\t:\t{type(self.uncertainty).__name__}(np.ndarray{self.uncertainty.shape})"
                    )
                except AttributeError:
                    print("\tuncertainty\t:\tUncertainty(None)")
            print()
            print("Time indices available: " + str(self.index.names))
            for key in self.index.names:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            if len(self._user_kwargs) == 0:
                return
            print()
            print("User defined attributes accessible via `object.key`")
            print("(displaying only unique values)")
            for key in self._user_kwargs:
                print(
                    f"\t{key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not defined')}"
                )

    @classmethod
    def from_pandas(cls, data, **kwargs):
        """Convert a pd.Series to a DataSeries"""
        return cls(data.to_numpy(), index=data.index, **kwargs)

    def stats_post_process(self, result, **kwargs):
        """Statistics post processer to format return data."""
        uncertainty = kwargs.pop("uncertainty", None)
        if uncertainty:
            return result, uncertainty
        else:
            return result


class DataSeries(MathMixin, StatsMixin, Series):
    """
    pandas.Series-like object with uncertainty and lightkurve functionality.

    pd.Series:
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, \\*, \\*\\*) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    data : array-like, Iterable
        Contains data stored in Series. If data is a dict, argument order is
        maintained.
    uncertainty : any type, optional
        Uncertainty in the dataset. [Not a standard part of pd.Series]
        Should have an attribute ``uncertainty_type`` that defines what kind of
        uncertainty is stored, for example ``"std"`` for standard deviation or
        ``"var"`` for variance. A metaclass defining such an interface is
        `NDUncertainty` - but isn't mandatory. If the uncertainty has no such
        attribute the uncertainty is stored as `StdDevm8Uncertainty`.
        Defaults to ``None``.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.

    Examples
    --------
    Constructing Series from a dictionary with an Index specified

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['a', 'b', 'c'])
    >>> ser
    a   1
    b   2
    c   3
    dtype: int64

    The keys of the dictionary match with the Index values, hence the Index
    values have no effect.

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> ser = pd.Series(data=d, index=['x', 'y', 'z'])
    >>> ser
    x   NaN
    y   NaN
    z   NaN
    dtype: float64

    Note that the Index is first build with the keys from the dictionary.
    After this the Series is reindexed with the given Index values, hence we
    get all NaN as a result.

    Constructing Series from a list with `copy=False`.

    >>> r = [1, 2]
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    [1, 2]
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `copy` of
    the original data even though `copy=False`, so
    the data is unchanged.

    Constructing Series from a 1d ndarray with `copy=False`.

    >>> r = np.array([1, 2])
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    array([999,   2])
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `view` on
    the original data, so
    the data is changed as well.
    """

    def __init__(self, data, uncertainty=None, index=None, dtype=None, **kwargs):
        self._metadata: List[str] = ["uncertainty"]
        self._user_kwargs: List[str] = []
        super().__init__(
            data,
            index=index,
            dtype=dtype,
            name=kwargs.pop("name", None),
            copy=kwargs.pop("copy", None),
            **kwargs,
        )
        self._array = self.to_numpy()
        self.uncertainty = uncertainty
        self._set_stats_methods()

    def __repr__(self):
        return f"📉 DataSeries {self.shape}"

    def _repr_html_(self):
        print(super().__repr__())


class BoolSeries(
    Series,
    BoolMixin,
):
    """
    pandas.Series-like object for bool datatypes with lightkurve functions.
    """

    def __init__(self, data, index=None, **kwargs):
        self._metadata: List[str] = []
        self._user_kwargs: List[str] = []
        kwargs.pop("dtype", None)
        super().__init__(
            data,
            index=index,
            dtype=bool,
            name=kwargs.pop("name", None),
            copy=kwargs.pop("copy", None),
            **kwargs,
        )

    def __repr__(self):
        return f"⚫️⚪️ BoolSeries {self.shape}"

    def _repr_html_(self):
        print(super().__repr__())


class BitwiseSeries(BitwiseMixin, Series):
    """
    pandas.Series-like object for bitwise datatypes with lightkurve functions.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. Entries are cast to the BitSet type in
        which integers are broken into their bits. I.e. 3 becomes {1, 2}.
    code_reference : dictionary, optional
        A dictionary of codes with keys that are powers of 2 and values are
        definitions of the code.
    values_display : str, default "int"
        What kind of display should be used? Must be one of "int", "bitset", or
        "detailed". "detailed" requires code_reference to  be provided, or else
        is identical to "parsed" which displays a set of the bits indicated by
        the binary representation of the integer value.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    """

    def __init__(
        self,
        data: Iterable[Union[Iterable[int], int]],
        code_dict: Dict = None,
        display_as: str = "int",
        index=None,
        **kwargs,
    ):
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
            data,
            index=index,
            dtype=object,
            name=kwargs.pop("name", None),
            copy=kwargs.pop("copy", None),
            **kwargs,
        )
        self.values_display = display_as
        self._user_kwargs.append("values_display")

    def __repr__(self):
        return f"📗 BitwiseSeries {self.shape}"

    def _repr_html_(self):
        if self._values_display == "detailed":
            display = repr(self.apply(lambda x: self.parse_code(x)))
        elif self._values_display == "bitset":
            display = repr(
                self.apply(
                    lambda x: str(self.breakdown(x)).replace("[", "{").replace("]", "}")
                )
            )
        else:
            display = repr(self.apply(int))
        return f"""<pre>{repr(self)}\n{display}</pre>"""
