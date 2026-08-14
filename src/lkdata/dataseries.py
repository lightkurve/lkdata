"""Classes and tools for working with 1 dimensional data."""

import logging
from abc import ABC
from typing import Dict, Iterable, List, Optional, Union, Any

import numpy as np
import pandas as pd

from .mixins import (
    AggMixin,
    ConvenienceMixins,
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
    """Abstract pd.Series-like extension class."""

    _pd_class = pd.Series
    _user_kwargs: Optional[List[str]] = None

    def __init__(
        self,
        data: Union[Iterable[Any], Dict],
        time_indices: Union[Dict, List, None] = None,
        **kwargs,
    ):
        time_indices = time_indices or {}

        # Pandas Series kwargs
        kwargs.pop("ntime", None)
        copy = kwargs.pop("copy", None)
        dtype = kwargs.pop("dtype", None)
        name = kwargs.pop("name", None)
        if not isinstance(data, dict):
            data = np.array(data)
        pdseries = pd.Series(data)
        index = kwargs.pop("index", None)
        if index is not None and not isinstance(index, pd.Index):
            index = pd.Index(index, name="given_index")
        index = self.parse_index(index, time_indices, pdseries.shape[0])

        """
        - data given as a dictionary will be allowed
        - a singular key with an array-like value will be treated as `name: data`, per intuition and akin to DataFrame objects
        - a longer dictionary will be treated like pandas Series
        - the index keyword argument will overwrite if the shapes don't match, otherwise it'll aggregate
        - I'll implement some cross matching  in the case of MultiIndex entries
        """
        if isinstance(data, dict):
            if len(data) == 1:
                name = name or next(
                    iter(data.keys())
                )  # name given as kwarg or inferred
                data = next(iter(data.values()))  # must be array-like
            elif index is not None and (len(index) != len(data)):
                # More than one key given, treating as comparison index
                check_index = list(data.keys())
                n_match = 0
                for level in index.levels:
                    common = set(level).intersection(check_index)
                    if len(common) > n_match:
                        inds_dict = np.where(np.isin(check_index, list(common)))
                        inds_index = np.where(np.isin(level, list(common)))
                        n_match = len(common)
                if n_match > 0:
                    data = np.array([*data.values()])[inds_dict]
                    index = index[inds_index]
            else:
                # Either index is given and shapes match, or no index was given
                # So use keys as a new index
                index = self.parse_index(index, {"key_index": list(data.keys())})
                data = [*data.values()]

        # User defined properties, stored as custom attributes
        self._user_kwargs = self._user_kwargs or []
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
            return self._build_instance(result_data, index=result_index, **init_kwds)

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
            if (
                hasattr(self, "uncertainty")
                and issubclass(self.__class__, DataSeries)
                and bool(self.uncertainty)
            ):
                print("Uncertainty:")
                try:
                    print(
                        f"  uncertainty\t:\t{type(self.uncertainty).__name__}(np.ndarray{self.uncertainty.shape})"
                    )
                except AttributeError:
                    print("  uncertainty\t:\tUncertainty(None)")
            print()
            print("Time indices available: " + str(self.index.names))
            for key in self.index.names:
                print(
                    f"  {key.ljust(max_name_len + 1)}:\t{getattr(self, key, 'Not Defined')}"
                )
            if len(self._user_kwargs) == 0:
                return
            print()
            print("User defined attributes accessible via `object.key`")
            for key in self._user_kwargs:
                print(
                    f"  {key.ljust(max_name_len + 1)} {type(getattr(self, key, None))}\t:\t{getattr(self, key, 'Not defined')}"
                )

    @classmethod
    def from_pandas(cls, data, **kwargs):
        """Convert a pd.Series to a DataSeries"""
        return cls(data.to_numpy(), index=data.index, **kwargs)

    def _stats_post_process(self, result, **kwargs):
        """Statistics post processer to format return data."""
        uncertainty = kwargs.pop("uncertainty", None)
        if uncertainty:
            return result, uncertainty
        else:
            return result


class DataSeries(StatsMixin, Series):
    """A pandas.Series-like object with uncertainty.

    Parameters
    ----------
    data : array-like, Iterable
        Contains data stored in Series. If data is a dict, argument order is
        maintained.
    uncertainty : array-like, optional
        Uncertainty in the dataset.
    index : array-like or Index (1d)
        Will default to RangeIndex (0, 1, 2, ..., n) if not provided.
        If data is dict-like and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
    """

    def __init__(
        self,
        data: Iterable[Any],
        uncertainty: Union[Iterable[Any], None] = None,
        index: Union[Iterable[Any], pd.MultiIndex, None] = None,
        dtype: Union[str, np.dtype, None] = None,
        **kwargs,
    ):
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
        return f"📉 DataSeries {self.shape}, Uncertainty: {bool(self.uncertainty)}"

    def _repr_html_(self):
        print(repr(self) + "\n" + super().__repr__())


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
        print(repr(self) + "/n" + super().__repr__())


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
        is identical to "bitset" which displays a set of the bits indicated by
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
        code_dict: Union[Dict, None] = None,
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
