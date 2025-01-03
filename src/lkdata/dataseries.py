"""Classes and tools for working with 1 dimensional data."""

import logging
from abc import ABC
from typing import Iterable, Union

import pandas as pd

from .mixins import (
    AggMixin,
    ConvenienceMixins,
    ErrorStatsMixin,
    MathMixin,
    StatsMixin,
    BoolStatsMixin,
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
    _user_kwargs = None

    def __init__(self, data, index=None, dtype=None, name=None, copy=None, **kwargs):
        self._user_kwargs = []
        time_indices = kwargs.pop("time_indices", None)
        index = self.parse_index(index, time_indices, data.shape[0])

        for key, val in kwargs.items():
            if key not in ("ntime", "index", "dtype"):
                self._user_kwargs.append(key)
                self._metadata.append(key)
                setattr(self, key, val)
        for key in self._user_kwargs:
            kwargs.pop(key)

        super().__init__(data, index=index, dtype=dtype, name=name, copy=copy)
        self.__post_init__()

    def __post_init__(self):
        def stats_post_process(result, **kwargs):
            return result

        self.stats_post_process = stats_post_process
        self._include_convenience_index()

    def __deepcopy__(self, *args, **kwargs):
        return self._build_instance(
            self.to_array(), index=self.index, **self.user_kwargs
        )

    @property
    def ntime(self):
        """Number of cadences in the data."""
        return self.shape[0]

    @classmethod
    def from_pandas(cls, data, **kwargs):
        """Convert a pd.Series to a DataSeries"""
        return cls(data.to_numpy(), index=data.index, **kwargs)

    def __getitem__(self, key):
        result_data = super().__getitem__(key)
        result_index = self.index[key]
        if isinstance(key, int):
            return result_data
        else:
            return self.__class__(result_data, index=result_index, **self.user_kwargs)


class DataSeries(Series, MathMixin, StatsMixin):
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
    data : array-like, Iterable, dict, or scalar value
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
        self.uncertainty = uncertainty
        super().__init__(
            data,
            index=index,
            dtype=dtype,
            name=kwargs.pop("name", None),
            copy=kwargs.pop("copy", None),
            **kwargs,
        )
        self._set_stats_methods()

    def __repr__(self):
        return f"📉 DataSeries {self.shape}\n" + super().__repr__()


class ErrorSeries(Series, ErrorStatsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_errstats_methods()

    def __repr__(self):
        return f"📈 ErrorSeries {self.shape}\n" + super().__repr__()


class BoolSeries(
    Series,
    BoolStatsMixin,
):
    """
    pandas.Series-like object for bool datatypes with lightkurve functions.
    """

    def __repr__(self):
        return f"⚫️⚪️ BoolSeries {self.shape}\n" + super().__repr__()


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
        What kind of display should be used? Must be one of "int", "parsed", or
        "detailed". "detailed" requires code_reference to  be provided, or else
        is identical to "parsed".
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
        values_display: str = "bitwise",
        index=None,
        **kwargs,
    ):
        # For pandas DataFrames subclasses, new properties must
        # be included in the _metadata list
        self._metadata = []
        self._user_kwargs = []
        kwargs["codes"] = kwargs.get("codes", {})
        self.codes = kwargs["codes"]
        super().__init__(
            data,
            index=index,
            dtype=object,
            name=kwargs.pop("name", None),
            copy=kwargs.pop("copy", None),
            **kwargs,
        )
        self.values_display = values_display
        self._user_kwargs.append("values_display")

    def __repr__(self):
        if self._values_display == "detailed":
            display = self.apply(lambda x: self.parse_code(x)).__repr__()
        elif self._values_display == "parsed":
            display = self.apply(
                lambda x: str(self.breakdown(x)).replace("[", "{").replace("]", "}")
            ).__repr__()
        else:
            display = self.apply(int).__repr__()
        return f"📗 BitwiseSeries {self.shape}\n" + display


class LkSeries:
    """A lightkurve class with Data, Error, Bool, and Bit Series.

    This product contains only Series products and supports all methods for
    a Series product, applying to all contained products.
    """

    ...
