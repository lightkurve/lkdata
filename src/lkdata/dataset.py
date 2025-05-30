"""Classes and tools for creating data bundles and batches"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatchmethod
from textwrap import dedent
from typing import Dict, Union
from warnings import warn

import numpy as np
import pandas as pd

from .datacube import Cube, DataCube, BoolCube, BitwiseCube
from .dataframe import Frame, DataFrame, BoolFrame, BitwiseFrame
from .dataseries import Series, DataSeries, BoolSeries, BitwiseSeries
from .mixins import IndexProcessorMixin

LkDataTypes = Union[DataCube, DataFrame, DataSeries]
LkBoolTypes = Union[BoolCube, BoolFrame, BoolSeries]
LkBitwiseTypes = Union[BitwiseCube, BitwiseFrame, BitwiseSeries]
LkTypes = Union[LkDataTypes, LkBoolTypes, LkBitwiseTypes]

CLS_STRINGS = {
    DataCube: "DataCube",
    BoolCube: "BoolCube",
    BitwiseCube: "BitwiseCube",
    DataFrame: "DataFrame",
    BoolFrame: "BoolFrame",
    BitwiseFrame: "BitwiseFrame",
    DataSeries: "DataSeries",
    BoolSeries: "BoolSeries",
    BitwiseSeries: "BitwiseSeries",
}


class DataProcessorMixin:
    """
    A mixin for processing and validating various types of lk data inputs.

    This class provides methods to process and validate input data, ensuring consistency
    with expected attributes and shapes. It supports various data types including
    DataCube, DataFrame, DataSeries.

    Attributes:
    -----------
    CLASS_CHECKS : dict
        A dictionary mapping data classes to sets of attributes to be checked.
    _data : dict
        Storage for processed data products.
    kwargs : dict or None
        Additional keyword arguments for data product construction.

    Methods:
    --------
    _check_attrs(data_product)
        Checks if the attributes of the data product match the expected values.
    _build_data_product(data_arr)
        Constructs the appropriate data product from the input array.
    process_input(data_input)
        Processes the input data and converts it to the appropriate data product.

    Notes:
    ------
    This class uses the singledispatchmethod decorator to handle different input types
    in the process_input method. Subclasses should implement any additional
    type-specific processing logic.
    """

    CLASS_CHECKS = {
        "cubes": {"ntime", "nrow", "ncol", "index", "columns"},
        "frames_series": {"ntime", "index"},
    }
    kwargs: dict = None

    def _check_attrs(self, data_product, name=""):
        if issubclass(type(data_product), Cube):
            attrs = self.CLASS_CHECKS["cubes"]
        else:
            attrs = self.CLASS_CHECKS["frames_series"]

        for attr in attrs.intersection({"ntime", "nrow", "ncol"}):
            if hasattr(self, attr) and (getattr(self, attr) is not None):
                if getattr(self, attr) != getattr(data_product, attr):
                    raise ValueError(
                        f"""
                        Mismatch: Dataset value for `{attr}` != `{attr}` for {name}
                        {getattr(self, attr)} != {getattr(data_product, attr)}
                        """
                    )
            else:
                setattr(self, attr, getattr(data_product, attr, None))
        # Checking indices/columns are the same length, not strict on values.
        for attr in attrs.intersection({"index", "columns"}):
            working_attr = getattr(self, attr, None)
            check_attr = getattr(data_product, attr, None)
            if working_attr is not None:
                if len(working_attr) != len(check_attr):
                    # attributes don't match and are incompatible based on length
                    raise ValueError(
                        f"""
                        Dataset shape for {attr} does not match given data {attr}.
                        {working_attr.shape} != {check_attr.shape}
                        """
                    )
                elif all(working_attr == check_attr):
                    # attributes match
                    pass
                elif attr == "colunmns":
                    # columns are indexed differently, but the same shape
                    # TODO: resolve this
                    pass
                elif attr == "index":
                    # attributes don't match, but indices are the same length and can be combined
                    new_index = pd.merge(
                        working_attr.to_frame().reset_index(drop=True),
                        check_attr.to_frame().reset_index(drop=True),
                        on="time_index",
                        how="inner",
                        suffixes=(None, f"_{name}"),
                    )
                    new_index = pd.MultiIndex.from_frame(new_index)

                    setattr(self, "index", new_index)
            else:
                # bundle does not have this property yet, assign it
                setattr(self, attr, check_attr)

    def _build_data_product(self, data_arr: Iterable):
        data_arr = np.asarray(data_arr)
        data_classes = {3: self._cube, 2: self._frame, 1: self._series}
        try:
            obj_class = data_classes.get(data_arr.ndim)
        except KeyError as err:
            raise ValueError(
                f"""
            The dimensions of given data ({data_arr.ndim=}) are not
            interpretable as a Cube, Frame, or Series.
            If giving multiple data products, provide input as a
            dictionary and use the `update` method.
                             """
            ) from err

        kwargs = deepcopy(self.kwargs)
        if data_arr.ndim in [1, 2]:
            # The following kwargs are assumed to be for building cubes when
            # initializing a DataSet and are ignored if a DataSet is built
            # with generic iterables
            kwargs.pop("columns", None)
            kwargs.pop("nrow", None)
            kwargs.pop("ncol", None)
            kwargs.pop("row_indices", None)
            kwargs.pop("col_indices", None)

        data_product = obj_class(data_arr, **kwargs)
        self._check_attrs(data_product)
        return data_product

    @singledispatchmethod
    def process_input(self, input_data) -> LkTypes:
        """
        Process the input data and convert it to the appropriate data product.

        This method uses single dispatch to handle different input types.

        Parameters
        ----------
        data_input : Union[list, np.ndarray, DataCube, DataFrame, DataSeries]
            The input data to be processed.
        Returns
        -------
        Union[DataCube, DataFrame, DataSeries]
            The processed data product.

        Raises
        ------
        TypeError
            If the input type is not supported or if multiple data products are
            provided without using a dictionary.
        ValueError
            If the data type is not recognized.
        """

    @process_input.register(DataCube)
    @process_input.register(DataFrame)
    @process_input.register(DataSeries)
    @process_input.register(BoolCube)
    @process_input.register(BoolFrame)
    @process_input.register(BoolSeries)
    @process_input.register(BitwiseCube)
    @process_input.register(BitwiseFrame)
    @process_input.register(BitwiseSeries)
    def _(self, input_data):
        self._check_attrs(input_data)
        return input_data

    @process_input.register
    def _(self, input_data: Iterable):
        try:
            data_product = self._build_data_product(input_data)
            return data_product
        except TypeError as err:
            raise TypeError(
                """
                If giving multiple data products, provide input as a
                dictionary and use the `update` method.
                """
            ) from err


class ProductBundle(dict, DataProcessorMixin):
    """A class to hold a collection of related data products.

    The products contained have the same basic attributes and share methods for
    slicing, aggregating, adding, downsampling, etc.

    All contained Cubes must have the same dimensions, all products must have
    the same time indices.

    Returns
    -------
    ProductBundle
        This class collects data and error products (cubes, frames, and/or series)
        as well as relevant metadata. Data products of the same type must have the
        same axes, i.e. time and pixel positions. Data aggregation methods
        managed by this DataSet are applied to all contained data products,
        i.e. down sampling. Data analysis should be performed on individual
        data products, i.e. flux summation.
    """

    _type: str = None
    _cube: Cube = Cube
    _frame: Frame = Frame
    _series: Series = Series

    _index: pd.MultiIndex = None
    _ntime: int = None

    def __init__(
        self,
        input_data: Union[
            Dict[str, Union[Iterable, LkDataTypes]],
            Iterable,
            LkTypes,
        ] = None,
        index: pd.MultiIndex = None,
        **kwargs,
    ):
        ntime = kwargs.pop("ntime", 0)
        time_indices = kwargs.pop("time_indices", None)
        index = IndexProcessorMixin.parse_index(index, time_indices, ntime)

        self._data_types = dict()
        if input_data is not None:
            input_data = self._unpack_input(input_data)
            self.update(input_data)
            for v in self.values():
                v.index = self.index

    def __deepcopy__(self, *args, **kwargs):
        return self.__class__({key: deepcopy(val) for key, val in self.items()})

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register
    def _(self, key: str):
        return super().__getitem__(key)

    @__getitem__.register(int)
    @__getitem__.register(slice)
    @__getitem__.register(tuple)
    @__getitem__.register(np.ndarray)
    def _(self, key):
        new_values = {}
        for k, v in self.items():
            try:
                new_values[k] = v[key]
            except IndexError:
                warn(f'Cannot parse {key} for "{k}" of type {v.__class__}', UserWarning)
                new_values[k] = v
        return new_values

    def __setitem__(self, key, val):
        # Check input attributes and convert to an LkType
        val = self.process_input(val)
        self._data_types[key] = type(val)
        super().__setitem__(key, val)
        if self.index is not None:
            setattr(val, "index", self.index)
        else:
            setattr(self, "index", val.index)
        if not self.ntime:
            setattr(self, "ntime", val.ntime)

    @singledispatchmethod
    def _unpack_input(self, input_data) -> dict:
        """Take given data and create a dictionary of products.

        Parameters
        ----------
        input : Union[Dict[str, Union[LkTypes, Iterable]], LkTypes, Iterable]
            Object or collection of objects to bundle with common indices. A
            dictionary of LkTypes|Iterable is the only option which supports
            creating a bundle of multiple objects. Lone LkTypes and Iterable
            objects are supported and will create a bundle with a single entry.

        Raises
        ------
        ValueError
            Raised if data is given in an unsupported form.
        """
        raise ValueError(f"Unsupported data type {type(input_data)}")

    @_unpack_input.register
    def _(self, input_data: dict):
        return input_data

    @_unpack_input.register(DataCube)
    @_unpack_input.register(DataFrame)
    @_unpack_input.register(DataSeries)
    @_unpack_input.register(BoolCube)
    @_unpack_input.register(BoolFrame)
    @_unpack_input.register(BoolSeries)
    @_unpack_input.register(BitwiseCube)
    @_unpack_input.register(BitwiseFrame)
    @_unpack_input.register(BitwiseSeries)
    def _(self, input_data):
        default_key = CLS_STRINGS.get(type(input_data))
        return {default_key: input_data}

    @_unpack_input.register
    def _(self, input_data: Iterable):
        data_as_array = np.asarray(input_data)
        product_type = {3: "Cube", 2: "Frame"}.get(data_as_array.ndim, "Series")
        return {self.type.capitalize() + product_type: data_as_array}

    def apply(self, func):
        mod = {key: func(val) for key, val in self.items()}
        return self.__class__(mod)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, pd.MultiIndex)
        self._index = value

    @property
    def ntime(self):
        return self._ntime

    @ntime.setter
    def ntime(self, value):
        assert isinstance(value, int)
        self._ntime = value

    @property
    def type(self):
        """The type of product in the bundle, data|bool|bitwise"""
        return self._type

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class DataProducts(ProductBundle):
    """
    A dict-like class for managing and processing data products.

    This class inherits from ProductBundle and is specifically designed to handle
    data (as opposed to error) products. It provides a container for various types
    of data arrays or data objects, with methods for processing and validating inputs.

    Parameters
    ----------
    data : Union[Dict, List, np.ndarray].
        The input data to be processed. Can be a dictionary of named data products,
        a list, or a numpy array.
    **kwargs
        Additional keyword arguments to be passed to the data product constructors.

    """

    _type = "data"
    _cube = DataCube
    _frame = DataFrame
    _series = DataSeries

    def __init__(
        self,
        data: Union[
            Dict[str, Union[Iterable, LkDataTypes]], LkDataTypes, Iterable
        ] = None,
        index: pd.MultiIndex = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(data, index)


class BoolProducts(ProductBundle):
    _type = "bool"
    _cube = BoolCube
    _frame = BoolFrame
    _series = BoolSeries

    def __init__(
        self,
        bools: Union[
            Dict[str, Union[Iterable, LkBoolTypes]], Iterable, LkBoolTypes
        ] = None,
        index: pd.MultiIndex = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(bools, index)


class BitwiseProducts(ProductBundle):
    _type = "bitwise"
    _cube = BitwiseCube
    _frame = BitwiseFrame
    _series = BitwiseSeries

    def __init__(
        self,
        bitwise: Union[
            Dict[str, Union[Iterable, LkBitwiseTypes]], Iterable, LkBitwiseTypes
        ] = None,
        index: pd.MultiIndex = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(bitwise, index)


@dataclass
class DataSet:
    """A class for objects with common time indices for batch manipulation.

    Parameters
    ----------
    data_products: Dict[str, LkDataTypes|Iterable]
        A dictionary of 1, 2, and/or 3 dimensional data objects. LkDataTypes
        support associated errors, whereas other Iterable types will be
        converted to the appropriate LkDataType without errors.
    bool_products: Dict[str, LkBoolTypes|Iterable[Bool]]
        A dictionary 1, 2, or 3 dimensional boolean arrays.
    bitwise_products: Dict[str, LkBitwiseTypes|Iterable[int|set|BitSet]]
        A dictionary of lkbitwise objects
    index: pd.MultiIndex, optional
       A MultiIndex which is used to index the data. If none given, the DataSet
       constructor will attempt to infer the index from the given products.
    time_indices: dict, optional

    Returns
    -------
    DataSet
        A dict-like object containing related data and error products which
        may be manipulated and analyzed simultaneously.

    Note: 2-dimensional arrays here are assumed to be a collection of
    time-series for a set of non-contiguous pixels rather than images.

    """

    _user_kwargs = None
    _index = None
    _ntime = None

    def __init__(
        self,
        data_products: Dict[str, Union[LkDataTypes, Iterable]] = None,
        bool_products: Dict[str, Union[LkBoolTypes, Iterable]] = None,
        bitwise_products: Dict[str, Union[LkBitwiseTypes, Iterable]] = None,
        index: pd.MultiIndex = None,
        time_indices: Dict[str, Iterable] = None,
        **kwargs,
    ):
        self._user_kwargs = []
        self.kwargs = kwargs
        # Custom keyword arguments given by the user.
        # They propagate to derivative products, but aren't used otherwise.
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k not in ("ntime", "nrow", "ncol", "columns"):
                self._user_kwargs.append(k)

        # WIP
        self.data_products = DataProducts(
            data_products, index=index, time_indices=time_indices, **kwargs
        )
        self.bool_products = BoolProducts(
            bool_products, index=index, time_indices=time_indices, **kwargs
        )
        self.bitwise_products = BitwiseProducts(
            bitwise_products, index=index, time_indices=time_indices, **kwargs
        )

        self.index = self._combine_indices(index=index)

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register
    def _(self, key: str):
        if key in self.contents:
            return self.contents[key]
        else:
            raise ValueError("Unrecognized key")

    @__getitem__.register(int)
    @__getitem__.register(list)
    @__getitem__.register(np.ndarray)
    def _(self, key):
        new_data = {}
        # new_data = dict(self.data)
        for data_key, data in self.data_products.items():
            new_data[data_key] = data[key]
        new_data = DataProducts(new_data, self.data_products.index[np.atleast_1d(key)])

        new_bool = {}
        for bool_key, bool_data in self.bool_products.items():
            new_bool[bool_key] = bool_data[key]
        new_bool = BoolProducts(
            new_bool, self.bitwise_products.index[np.atleast_1d(key)]
        )

        new_bit = {}
        for bit_key, bit_data in self.bitwise_products.items():
            new_bit[bit_key] = bit_data[key]
        new_bit = BitwiseProducts(
            new_bit, self.bitwise_products.index[np.atleast_1d(key)]
        )

        return self._build_instance(
            newdata=new_data, newbools=new_bool, newbits=new_bit
        )

    @__getitem__.register(slice)
    def _(self, key):
        new_data = {}
        for data_key, data in self.data_products.items():
            new_data[data_key] = data[key]
        new_data = DataProducts(new_data, self.data_products.index[key])

        new_bool = {}
        for bool_key, bool_data in self.bool_products.items():
            new_bool[bool_key] = bool_data[key]
        new_bool = BoolProducts(new_bool, self.bitwise_products.index[key])

        new_bit = {}
        for bit_key, bit_data in self.bitwise_products.items():
            new_bit[bit_key] = bit_data[key]
        new_bit = BitwiseProducts(new_bit, self.bitwise_products.index[key])

        return self._build_instance(
            newdata=new_data, newbools=new_bool, newbits=new_bit
        )

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        if len(key) not in [1, 2, 3]:
            raise KeyError(f"Cannot parse key with {len(key)} elements.")

        new_data = {
            data_key: data[key]
            for data_key, data in self.data_products.items()
            if isinstance(data, DataCube)
        }

        if len(new_data) > 0:
            new_columns = list(new_data.values())[0].columns
        else:
            new_columns = pd.MultiIndex.from_arrays([[]], names=["series"])

        # Just slicing/selecting on time for Series and Frames
        new_data.update(
            {
                data_key: data[time_key]
                for data_key, data in self.data_products.items()
                if isinstance(data, (DataSeries, DataFrame))
            }
        )

        if len(new_data) > 0:
            new_index = list(new_data.values())[0].index
        else:
            new_index = pd.MultiIndex.from_arrays([[]], names=["time_index"])

        new_bool = {
            data_key: data[key]
            for data_key, data in self.bool_products.items()
            if isinstance(data, BoolCube)
        }

        if len(new_bool) > 0:
            new_columns = list(new_bool.values())[0].columns
        else:
            new_columns = pd.MultiIndex.from_arrays([[]], names=["series"])

        # Just slicing/selecting on time for Series and Frames
        new_bool.update(
            {
                data_key: data[time_key]
                for data_key, data in self.bool_products.items()
                if isinstance(data, (BoolSeries, BoolFrame))
            }
        )

        new_bit = {
            data_key: data[key]
            for data_key, data in self.bitwise_products.items()
            if isinstance(data, BitwiseCube)
        }

        if len(new_bit) > 0:
            new_columns = list(new_bit.values())[0].columns
        else:
            new_columns = pd.MultiIndex.from_arrays([[]], names=["series"])

        # Just slicing/selecting on time for Series and Frames
        new_bit.update(
            {
                data_key: data[time_key]
                for data_key, data in self.bitwise_products.items()
                if isinstance(data, (BitwiseSeries, BitwiseFrame))
            }
        )

        if len(new_data) > 0:
            new_index = list(new_data.values())[0].index
        else:
            new_index = pd.MultiIndex.from_arrays([[]], names=["time_index"])

        new_kwargs = self.kwargs.copy()
        new_kwargs["index"] = new_index
        new_kwargs["columns"] = new_columns

        return self._build_instance(
            new_data, newbools=new_bool, newbits=new_bit, **new_kwargs
        )

    def __len__(self):
        return self.ntime

    def __repr__(self):
        msg = f"""
        Data Products:
        {self.data_products}
        Bool Products:
        {self.bool_products}
        Bitwise Products:
        {self.bitwise_products}
        Properties:
        {list(self.kwargs.keys())}
        """
        msg = dedent(msg)
        return msg

    def _attr_override(self, attr, val):
        setattr(self.data_products, attr, val)
        for v in self.data_products.values():
            setattr(v, attr, val)
        setattr(self.bool_products, attr, val)
        for v in self.bool_products.values():
            setattr(v, attr, val)
        setattr(self.bitwise_products, attr, val)
        for v in self.bitwise_products.values():
            setattr(v, attr, val)

    def _batch_wrapper(self, func):
        def batch_func(*args, **kwargs):
            def do_batch_func(bundle):
                for key, val in bundle.items():
                    obj_func = getattr(val, func)
                    bundle[key] = obj_func(*args, **kwargs)
                return bundle

            newdata = do_batch_func(dict(deepcopy(self.data_products)))
            newbools = do_batch_func(dict(deepcopy(self.bool_products)))
            newbits = do_batch_func(dict(deepcopy(self.bitwise_products)))

            return self._build_instance(newdata, newbools, newbits)

        return batch_func

    def _build_instance(self, newdata, newbools, newbits, **kwargs):
        all_kwargs = self.user_kwargs.copy()
        all_kwargs.update(**kwargs)
        return self.__class__(newdata, newbools, newbits, **all_kwargs)

    def _combine_indices(self, index) -> pd.MultiIndex:
        """Combine indices of contents into a single MultiIndex."""

        def combine(working_index, check_index, suffix=""):
            if check_index is not None and working_index is not None:
                if len(check_index) != len(working_index):
                    raise ValueError(
                        "Length of BoolProducts index"
                        f"({len(check_index)})"
                        "does not match the length of DataSet index"
                        f"({len(working_index)})"
                    )
                else:
                    index_df1 = working_index.to_frame().reset_index(drop=True)
                    index_df2 = check_index.to_frame().reset_index(drop=True)
                    common_cols = index_df1.columns.intersection(index_df2.columns)
                    matching_cols = common_cols[
                        (index_df1[common_cols] == index_df2[common_cols]).all()
                    ]
                    new_index = pd.merge(
                        index_df1,
                        index_df2,
                        on=list(matching_cols),
                        how="inner",
                        suffixes=(None, suffix),
                    )
                    new_index = pd.MultiIndex.from_frame(new_index)
                    working_index = new_index
            elif working_index is None:
                working_index = check_index
            return working_index

        working_index = index
        working_index = combine(working_index, self.data_products.index, "_data")
        working_index = combine(working_index, self.bool_products.index, "_bool")
        working_index = combine(working_index, self.bitwise_products.index, "_bitwise")
        self.data_products.index = working_index
        self.bool_products.index = working_index
        self.bitwise_products.index = working_index
        return working_index

    @property
    def cubes(self) -> dict:
        """Retrieve all Cube objects.

        Returns
        -------
        dict
            A dictionary containing all Cube objects from the DataSet.
            The keys are the original keys, given or generated.
        """
        cubes = {
            key: value
            for key, value in self.contents.items()
            if issubclass(type(value), Cube)
        }

        return cubes

    @property
    def frames(self) -> dict:
        """Retrieve all Frame objects.

        Returns
        -------
        dict
            A dictionary containing all Frame objects from the DataSet.
            The keys are the original keys, given or generated.
        """
        frames = {
            key: value
            for key, value in self.contents.items()
            if issubclass(type(value), Frame)
        }
        return frames

    @property
    def index(self) -> pd.MultiIndex:
        return self._index

    @index.setter
    def index(self, val: pd.MultiIndex):
        self._attr_override("index", val)
        self._index = val
        self._ntime = len(val)

    @property
    def ntime(self) -> int:
        return self._ntime

    @ntime.setter
    def ntime(self, val):
        if self.index.empty:
            self._ntime = val
            self._attr_override("ntime", val)
            new_index = DataCube.parse_index(ntime=val)
            self._attr_override("index", new_index)
        else:
            raise AttributeError("Cannot set ntime when a non-empty index exists.")

    @property
    def contents(self) -> dict:
        """All contained lkdata objects"""
        contents = {}
        contents.update(self.data_products)
        contents.update(self.bool_products)
        contents.update(self.bitwise_products)
        return contents

    @property
    def series(self) -> dict:
        """Retrieve all Series objects.

        Returns
        -------
        dict
            A dictionary containing all Series objects from the DataSet.
            The keys are the original keys, given or generated.
        """
        series = {
            key: value
            for key, value in self.contents.items()
            if issubclass(type(value), Series)
        }
        return series

    @property
    def user_kwargs(self) -> dict:
        """Keywords passed by the user"""
        return {key: getattr(self, key, None) for key in self._user_kwargs}

    def downsample(self, nframes: int = 5, level: Union[str, int] = -1):
        """Downsample all contained products."""
        downsample = self._batch_wrapper("downsample")
        return downsample(nframes, level)

    def droplevel(self, level, axis=0):
        """Drop a given level from the index."""
        droplevel = self._batch_wrapper("droplevel")
        return droplevel(level, axis)

    def fold(
        self,
        period: float,
        t0: float = None,
        level: Union[int, str] = -1,
        inplace: bool = False,
        label: str = "phase",
    ):
        """Phase fold all data products."""
        if period <= 0:
            raise ValueError("`period` must be greater than 0.")
        index = deepcopy(self.index)
        if len(self.index.names) == 1:
            # Cadence is typically level 0 and datetimes levels 1+
            level = 0

        if label in index.names:
            index = index.droplevel(label)

        time = index.get_level_values(level)
        if t0:
            time = time - t0
        else:
            time = time - time.min()

        phase = time % period / period
        indices = index.to_frame()
        indices[label] = phase
        indices.set_index(label, append=True, inplace=True)

        if inplace:
            newbatch = self
        else:
            newbatch = deepcopy(self)

        newbatch.index = indices.index
        setattr(newbatch, label, newbatch.index.get_level_values(level=label))
        for val in newbatch.data_products.values():
            val.index = indices.index

        if not inplace:
            return newbatch
