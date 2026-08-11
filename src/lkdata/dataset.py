"""Classes and tools for creating data bundles and batches"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Dict, Union, Type
from warnings import warn

import numpy as np
import pandas as pd

from .datacube import Cube, DataCube, BoolCube, BitwiseCube
from .seriescollection import (
    SeriesCollection,
    DataSeriesCollection,
    BoolSeriesCollection,
    BitwiseSeriesCollection,
)
from .dataseries import Series, DataSeries, BoolSeries, BitwiseSeries
from .mixins import IndexProcessorMixin

LkDataTypes = Union[DataCube, DataSeriesCollection, DataSeries]
LkBoolTypes = Union[BoolCube, BoolSeriesCollection, BoolSeries]
LkBitwiseTypes = Union[BitwiseCube, BitwiseSeriesCollection, BitwiseSeries]
LkTypes = Union[LkDataTypes, LkBoolTypes, LkBitwiseTypes]

CLS_STRINGS = {
    DataCube: "DataCube",
    BoolCube: "BoolCube",
    BitwiseCube: "BitwiseCube",
    DataSeriesCollection: "DataSeriesCollection",
    BoolSeriesCollection: "BoolSeriesCollection",
    BitwiseSeriesCollection: "BitwiseSeriesCollection",
    DataSeries: "DataSeries",
    BoolSeries: "BoolSeries",
    BitwiseSeries: "BitwiseSeries",
}


class DataProcessorMixin(ABC):
    """
    A mixin for processing and validating various types of lk data inputs.

    This class provides methods to process and validate input data, ensuring consistency
    with expected attributes and shapes. It supports various data types including
    DataCube, DataSeriesCollection, DataSeries.

    Attributes:
    -----------
    CLASS_CHECKS : dict
        A dictionary mapping data classes to sets of attributes to be checked.
    kwargs : dict or None
        Additional keyword arguments for data product construction.

    Methods:
    --------
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
        "seriescollections_series": {"ntime", "index"},
    }

    @property
    @abstractmethod
    def _cube(self):
        pass

    @property
    @abstractmethod
    def _seriescollection(self):
        pass

    @property
    @abstractmethod
    def _series(self):
        pass

    def _check_attrs(self, data_product: LkTypes, name=""):
        """Checks if the attributes of the data product match the expected values.

        Parameters
        ----------
        data_product : LkTypes
            The data product to check attributes for.
        name : str, optional
            Name of the data product, used for error messages.

        Raises
        ------
        ValueError
            If there's a mismatch between the dataset attributes and the data product attributes.
        """
        if issubclass(type(data_product), Cube):
            attrs = self.CLASS_CHECKS["cubes"]
        else:
            attrs = self.CLASS_CHECKS["seriescollections_series"]

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
            if working_attr is not None and check_attr is not None:
                if len(working_attr) != len(check_attr):
                    # attributes don't match and are incompatible based on length
                    raise ValueError(
                        f"""
                        Dataset shape for {attr} does not match given data {attr}.
                        {working_attr.shape} != {check_attr.shape}
                        """
                    )
                elif working_attr.equals(check_attr):
                    # attributes match
                    pass
                elif attr == "colunmns":
                    # columns are indexed differently, but the same shape
                    working_column = working_attr.to_frame().reset_index(drop=True)
                    working_column["merge"] = working_column.index
                    check_column = check_attr.to_frame().reset_index(drop=True)
                    check_column["merge"] = working_column.index
                    new_columns = pd.merge(
                        working_column,
                        check_column,
                        on="merge",
                        how="inner",
                        suffixes=(None, f"_{name}"),
                    )
                    new_columns = pd.MultiIndex.from_frame(new_columns)
                    setattr(self, "columns", new_columns)
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
            elif working_attr is None and check_attr is not None:
                # bundle does not have this property yet, assign it
                setattr(self, attr, check_attr)

    def _build_data_product(self, data_arr: Iterable):
        """Constructs the appropriate data product from the input array."""
        data_arr = np.asarray(data_arr)
        data_classes = {3: self._cube, 2: self._seriescollection, 1: self._series}
        try:
            obj_class = data_classes.get(data_arr.ndim)
        except KeyError as err:
            raise ValueError(
                f"""
            The dimensions of given data ({data_arr.ndim=}) are not
            interpretable as a Cube, SeriesCollection, or Series.
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
        data_input : Union[list, np.ndarray, DataCube, DataSeriesCollection, DataSeries]
            The input data to be processed.
        Returns
        -------
        Union[DataCube, DataSeriesCollection, DataSeries]
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
    @process_input.register(DataSeriesCollection)
    @process_input.register(DataSeries)
    @process_input.register(BoolCube)
    @process_input.register(BoolSeriesCollection)
    @process_input.register(BoolSeries)
    @process_input.register(BitwiseCube)
    @process_input.register(BitwiseSeriesCollection)
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
        This class collects data and error products (cubes, seriescollections, and/or series)
        as well as relevant metadata. Data products of the same type must have the
        same axes, i.e. time and pixel positions. Data aggregation methods
        managed by this DataSet are applied to all contained data products,
        i.e. down sampling. Data analysis should be performed on individual
        data products, i.e. flux summation.
    """

    _type: str = None
    _cube: Type[Cube] = Cube
    _seriescollection: Type[SeriesCollection] = SeriesCollection
    _series: Type[Series] = Series

    _index: Union[pd.MultiIndex, None] = None
    _ntime: Union[int, None] = None

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
        if (index is not None) or (time_indices is not None) or (ntime != 0):
            index = IndexProcessorMixin.parse_index(index, time_indices, ntime)
            self.index = index
        self._data_types = dict()
        if input_data is not None:
            input_data = self._unpack_input(input_data)
            self.update(input_data)

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
            val._include_convenience_index()
        else:
            setattr(self, "index", val.index)
        if not self.ntime:
            setattr(self, "ntime", val.ntime)

    def set_attr(self, attr, val):
        setattr(self, attr, val)
        for v in self.values():
            setattr(v, attr, val)

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
    @_unpack_input.register(DataSeriesCollection)
    @_unpack_input.register(DataSeries)
    @_unpack_input.register(BoolCube)
    @_unpack_input.register(BoolSeriesCollection)
    @_unpack_input.register(BoolSeries)
    @_unpack_input.register(BitwiseCube)
    @_unpack_input.register(BitwiseSeriesCollection)
    @_unpack_input.register(BitwiseSeries)
    def _(self, input_data):
        default_key = CLS_STRINGS.get(type(input_data))
        return {default_key: input_data}

    @_unpack_input.register
    def _(self, input_data: Iterable):
        data_as_array = np.asarray(input_data)
        product_type = {3: "Cube", 2: "SeriesCollection"}.get(
            data_as_array.ndim, "Series"
        )
        return {self.type.capitalize() + product_type: data_as_array}

    def apply(self, func):
        mod = {key: func(val) for key, val in self.items()}
        return self.__class__(mod)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, val):
        assert isinstance(val, pd.MultiIndex)
        self._index = val

    @property
    def ntime(self):
        return self._ntime

    @ntime.setter
    def ntime(self, val):
        assert isinstance(val, int)
        self._ntime = val

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
    _seriescollection = DataSeriesCollection
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
        super().__init__(data, index, **kwargs)


class BoolProducts(ProductBundle):
    _type = "bool"
    _cube = BoolCube
    _seriescollection = BoolSeriesCollection
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
        super().__init__(bools, index, **kwargs)


class BitwiseProducts(ProductBundle):
    _type = "bitwise"
    _cube = BitwiseCube
    _seriescollection = BitwiseSeriesCollection
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
        super().__init__(bitwise, index, **kwargs)


class AttrsHolder(dict):
    _parent = None

    def __init__(self, input, parent):
        self._parent = parent
        super().__init__(input)

    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        setattr(self._parent, key, val)
        self._parent._attr_override(key, val)


@dataclass
class DataSet:
    """A class for objects with common time indices for batch manipulation.

    This is a container for products related by time and space. Every contained
    product shares the same time indices and all Cubes share spatial indices.
    The contained products may fall into three categories: data (plus errors),
    boolean, and bitwise. An example of data would be flux, boolean may contain
    aperture masks, and bitwise could contain error codes.

    Note: 2-dimensional arrays here are assumed to be a collection of
    time-series for a set of non-contiguous pixels rather than images.

    """

    _index = None
    _ntime = 0
    _data_products = None
    _bool_products = None
    _bitwise_products = None

    def __init__(
        self,
        data_products: Union[None, Dict[str, Union[LkDataTypes, Iterable]]] = None,
        bool_products: Union[None, Dict[str, Union[LkBoolTypes, Iterable]]] = None,
        bitwise_products: Union[
            None, Dict[str, Union[LkBitwiseTypes, Iterable]]
        ] = None,
        index: Union[None, pd.MultiIndex] = None,
        time_indices: Union[None, Dict[str, Iterable]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data_products: Dict[str, LkDataTypes|Iterable], optional
            A dictionary of 1, 2, and/or 3 dimensional data objects. LkDataTypes
            support associated errors, whereas other Iterable types will be
            converted to the appropriate LkDataType without errors.
        bool_products: Dict[str, LkBoolTypes|Iterable[Bool]], optional
            A dictionary 1, 2, or 3 dimensional boolean arrays.
        bitwise_products: Dict[str, LkBitwiseTypes|Iterable[int|set|BitSet]], optional
            A dictionary of lkbitwise objects
        index: pd.MultiIndex, optional
            A pandas MultiIndex object containing the times for all contained
            products. Combined with `time_indices` if both are given.
        time_indices: Dict[str, Iterable], optional
            A dictionary of time indices which will be converted into or combined
            with the `index`.

        A MultiIndex which is used to index the data. If none given, the DataSet
        constructor will attempt to infer the index from the given products.
        time_indices: dict, optional

        """
        self._attrs = AttrsHolder({}, self)
        self.kwargs = kwargs
        # Custom keyword arguments given by the user.
        # They propagate to derivative products, but aren't used otherwise.

        for k, v in kwargs.items():
            self._attrs[k] = v

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
        """Like a dictionary retreival."""
        if key in self.contents:
            return self.contents[key]
        else:
            raise ValueError("Unrecognized key")

    @__getitem__.register(int)
    @__getitem__.register(Iterable)
    def _(self, key):
        """Time index subselection

        returns a new DataSet with the indices given
        """
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
        """Time index subselection

        returns a new DataSet with the time slice
        """
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
        """Time and space slice/selection. Spatial cut only applies to Cubes.

        returns a new DataSet with the cut applied.
        """
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

        # Just slicing/selecting on time for Series and SeriesCollections
        new_data.update(
            {
                data_key: data[time_key]
                for data_key, data in self.data_products.items()
                if isinstance(data, (DataSeries, DataSeriesCollection))
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

        # Just slicing/selecting on time for Series and SeriesCollections
        new_bool.update(
            {
                data_key: data[time_key]
                for data_key, data in self.bool_products.items()
                if isinstance(data, (BoolSeries, BoolSeriesCollection))
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

        # Just slicing/selecting on time for Series and SeriesCollections
        new_bit.update(
            {
                data_key: data[time_key]
                for data_key, data in self.bitwise_products.items()
                if isinstance(data, (BitwiseSeries, BitwiseSeriesCollection))
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

    def __setitem__(self, key, val):
        if isinstance(val, LkDataTypes.__args__):
            self.data_products[key] = val
        elif isinstance(val, LkBoolTypes.__args__):
            self.bool_products[key] = val
        elif isinstance(val, LkBitwiseTypes.__args__):
            self.bitwise_products[key] = val
        else:
            raise TypeError(
                f"Type must be one of {LkTypes} when assigning directly by key."
            )

    def __len__(self):
        return self.ntime

    def __repr__(self):
        msg = f"🗂️ DataSet: {len(self.contents)} product(s)."
        if len(self.data_products) > 0:
            msg += f"\nData Products:\n  {self.data_products}"
        if len(self.bool_products) > 0:
            msg += f"\nBool Products:\n  {self.bool_products}"
        if len(self.bitwise_products) > 0:
            msg += f"\nBitwise Products:\n  {self.bitwise_products}"
        if len(self.kwargs) > 0:
            msg += f"\nProperties:\n  {list(self.kwargs.keys())}"
        return msg

    def _attr_override(self, attr, val):
        if self.data_products is not None:
            self.data_products.set_attr(attr, val)
        if self.bool_products is not None:
            self.bool_products.set_attr(attr, val)
        if self.bitwise_products is not None:
            self.bitwise_products.set_attr(attr, val)

    def _batch_wrapper(self, func: str, cubes_only: bool = False):
        def batch_func(*args, **kwargs):
            def do_batch_func(bundle):
                for key, val in bundle.items():
                    obj_func = getattr(val, func)
                    bundle[key] = obj_func(*args, **kwargs)
                return bundle

            if cubes_only:
                datacubes = {
                    key: deepcopy(val)
                    for key, val in self.data_products.items()
                    if issubclass(val.__class__, Cube)
                }
                newdata = do_batch_func(datacubes)
                newdata.update(
                    {
                        key: deepcopy(val)
                        for key, val in self.data_products.items()
                        if not issubclass(val.__class__, Cube)
                    }
                )

                boolcubes = {
                    key: deepcopy(val)
                    for key, val in self.bool_products.items()
                    if issubclass(val.__class__, Cube)
                }
                newbools = do_batch_func(boolcubes)
                newbools.update(
                    {
                        key: deepcopy(val)
                        for key, val in self.bool_products.items()
                        if not issubclass(val.__class__, Cube)
                    }
                )

                bitcubes = {
                    key: deepcopy(val)
                    for key, val in self.bitwise_products.items()
                    if issubclass(val.__class__, Cube)
                }
                newbits = do_batch_func(bitcubes)
                newbits.update(
                    {
                        key: deepcopy(val)
                        for key, val in self.bitwise_products.items()
                        if not issubclass(val.__class__, Cube)
                    }
                )

            else:
                newdata = do_batch_func(dict(deepcopy(self.data_products)))
                newbools = do_batch_func(dict(deepcopy(self.bool_products)))
                newbits = do_batch_func(dict(deepcopy(self.bitwise_products)))

            return self._build_instance(newdata, newbools, newbits)

        return batch_func

    def _build_instance(self, newdata, newbools, newbits, **kwargs):
        all_kwargs = self.attrs.copy()
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
                        (
                            index_df1[common_cols].dropna()
                            == index_df2[common_cols].dropna()
                        ).all()
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
            key: val
            for key, val in self.contents.items()
            if issubclass(type(val), Cube)
        }

        return cubes

    @property
    def data_products(self):
        return self._data_products

    @data_products.setter
    def data_products(self, val: DataProducts):
        self._data_products = val

    @property
    def bool_products(self):
        return self._bool_products

    @bool_products.setter
    def bool_products(self, val: BoolProducts):
        self._bool_products = val

    @property
    def bitwise_products(self):
        return self._bitwise_products

    @bitwise_products.setter
    def bitwise_products(self, val: BitwiseProducts):
        self._bitwise_products = val

    @property
    def series_collections(self) -> dict:
        """Retrieve all SeriesCollection objects.

        Returns
        -------
        dict
            A dictionary containing all SeriesCollection objects from the DataSet.
            The keys are the original keys, given or generated.
        """
        seriescollections = {
            key: val
            for key, val in self.contents.items()
            if issubclass(type(val), SeriesCollection)
        }
        return seriescollections

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
            key: val
            for key, val in self.contents.items()
            if issubclass(type(val), Series)
        }
        return series

    @property
    def attrs(self) -> AttrsHolder:
        """Keywords passed by the user"""
        return self._attrs

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

    def spatial_downsample(self, *args, **kwargs):
        s_downsample = self._batch_wrapper("spatial_downsample", cubes_only=True)
        return s_downsample(*args, **kwargs)

    def describe_set(self, **printoptions):
        """Print a description of the DataSet instance.

        This description prints information about the temporal
        indices available in the DataSet. It also prints out any additional
        user-assigned properties.
        """
        printoptions["linewidth"] = printoptions.get("linewidth", 79)
        printoptions["edgeitems"] = printoptions.get("edgeitems", 2)
        printoptions["threshold"] = printoptions.get("threshold", 20)
        with np.printoptions(**printoptions):
            print(f"🗂️ DataSet: {len(self.contents)} product(s).\n")
            if len(self.data_products) > 0:
                print(f"\nData Products:\n  {self.data_products}\n")
            if len(self.bool_products) > 0:
                print(f"\nBool Products:\n  {self.bool_products}\n")
            if len(self.bitwise_products) > 0:
                print(f"\nBitwise Products:\n  {self.bitwise_products}\n")
            print("Time indices available: " + str(self.index.names))
            print()
            if len(self.attrs) == 0:
                return
            max_name_len = max(map(len, self.attrs))
            print("User defined attributes accessible via `object.key`")
            for key, val in self.attrs.items():
                print(
                    f"  {key.ljust(max_name_len + 1)} {type(getattr(self, key, None))}\t:\t{getattr(self, key, 'Not defined')}"
                )
