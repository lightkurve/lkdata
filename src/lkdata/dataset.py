"""Classes and tools for creating data bundles and batches"""
from collections.abc import Iterable
from copy import deepcopy
from functools import singledispatchmethod
from textwrap import dedent
from typing import Dict, Union
from warnings import warn

import numpy as np
import pandas as pd

from .datacube import DataCube, BoolCube, BitwiseCube
from .dataframe import DataFrame, BoolFrame, BitwiseFrame
from .dataseries import DataSeries, BoolSeries, BitwiseSeries

LkDataTypes = Union[DataCube, DataFrame, DataSeries]
LkBoolTypes = Union[BoolCube, BoolFrame, BoolSeries]
LkBitwiseTypes = Union[BitwiseCube, BitwiseFrame, BitwiseSeries]
LkTypes = Union[LkDataTypes, LkBoolTypes, LkBitwiseTypes]


class DataProcessorMixin:
    """
    A mixin for processing and validating various types of lk data inputs.

    This class provides methods to process and validate input data, ensuring consistency
    with expected attributes and shapes. It supports various data types including
    DataCube, ErrorCube, DataFrame, ErrorFrame, DataSeries, and ErrorSeries.

    Attributes:
    -----------
    CLASS_CHECKS : dict
        A dictionary mapping data classes to sets of attributes to be checked.
    _data : dict
        Storage for processed data products.
    _error : dict
        Storage for processed error products.
    _type : str or None
        The type of data being processed ('data' or 'error').
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
        DataCube: {"ntime", "nrow", "ncol", "index", "columns"},
        DataFrame: {"ntime", "index"},
        DataSeries: {"ntime", "index"},
    }
    _data = {}
    _error = {}
    _type = None
    kwargs = None

    def _check_attrs(self, data_product):
        attrs = self.CLASS_CHECKS[data_product.__class__]
        for attr in attrs.intersection({"ntime", "nrow", "ncol"}):
            if hasattr(self, attr):
                if getattr(self, attr) != getattr(data_product, attr):
                    raise ValueError(
                        f"""
                        Dataset value for {attr} != given data {attr}
                        {getattr(self, attr)} != {getattr(data_product, attr)}
                        """
                    )
            else:
                setattr(self, attr, getattr(data_product, attr, None))
        for attr in attrs.intersection({"index", "columns"}):
            if hasattr(self, attr):
                if getattr(self, attr).shape != getattr(data_product, attr).shape:
                    raise ValueError(
                        f"""
                        Dataset shape for {attr} does not match given data {attr}.
                        {getattr(self, attr).shape} != {getattr(data_product, attr).shape}
                        """
                    )
            else:
                setattr(self, attr, getattr(data_product, attr, None))

    def _build_data_product(self, data_arr: Iterable):
        data_arr = np.asarray(data_arr)
        data_classes = {3: DataCube, 2: DataFrame, 1: DataSeries}
        classes = {"data": data_classes}

        obj_class = classes[self._type].get(data_arr.ndim, None)
        if obj_class in [DataCube]:
            data_product = obj_class(data_arr, **self.kwargs)
            self._check_attrs(data_product)
        elif obj_class:
            kwargs = deepcopy(self.kwargs)
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
        else:
            raise TypeError("Unrecognized format given for input.")
        return data_product

    @singledispatchmethod
    def process_input(self, data_input) -> LkTypes:
        """
        Process the input data and convert it to the appropriate data product.

        This method uses single dispatch to handle different input types.

        Parameters
        ----------
        data_input : Union[list, np.ndarray, DataCube, DataFrame, DataSeries,
                           ErrorCube, ErrorFrame, ErrorSeries]
            The input data to be processed.
        Returns
        -------
        Union[DataCube, DataFrame, DataSeries, ErrorCube, ErrorFrame, ErrorSeries]
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
    def _(self, data_input):
        self._check_attrs(data_input)
        return data_input

    @process_input.register
    def _(self, data_input: Iterable):
        assert self._type in ["data", "error"], "Unrecognized type"
        try:
            data_product = self._build_data_product(data_input)
            return data_product
        except TypeError as err:
            raise TypeError(
                """
                If giving multiple data products, provide input as a
                dictionary and use the `update` method."""
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

    _type = None

    def __init__(
        self,
        data: Union[
            Dict[str, Union[Iterable, LkDataTypes]], Iterable, LkDataTypes
        ] = None,
        index: pd.MultiIndex = None,
    ):
        if isinstance(index, pd.MultiIndex):
            self.index = index
        self._data_types = dict()
        if data is not None:
            data = self._unpack_data(data)
            self.update(data)

    @singledispatchmethod
    def _unpack_data(self, data):
        raise ValueError(f"Unsupported data type {type(data)}")

    @_unpack_data.register
    def _(self, data: dict):
        return data

    @_unpack_data.register(DataCube)
    @_unpack_data.register(DataFrame)
    @_unpack_data.register(DataSeries)
    def _(self, data):
        return {"flux_" + self._type: data}

    @_unpack_data.register
    def _(self, data: Iterable):
        return {"flux_" + self._type: np.asarray(data)}

    def __setitem__(self, key, val):
        val = self.process_input(val)
        self._data_types[key] = type(val)
        dict.__setitem__(self, key, val)
        if not hasattr(self, "index"):
            setattr(self, "index", val.index)
        if not hasattr(self, "ntime"):
            setattr(self, "ntime", val.ntime)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def apply(self, func):
        mod = {key: func(val) for key, val in self.items()}
        return self.__class__(mod)

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

    def __deepcopy__(self, *args, **kwargs):
        return self.__class__({key: deepcopy(val) for key, val in self.items()})


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


class DataSet:
    """Class to group related data and error products for batch processing.

    Parameters
    ----------
    data: Union[Iterable, lkDataTypes, DataProducts, Dict[str, Union[Iterable, lkDataTypes]]]
        Data which sums linearly. Providing a dictionary
        of lightkurve data products (DataCube, DataFrame, DataSeries) is recommended.
        However, a singular array-like or a dictionary of array-like data may
        be given and will be converted into the corresponding lkDataType based
        on the shape of the data.

    error: Union[Iterable, lkErrorTypes, ErrorProducts, Dict[str, Union[Iterable, lkErrorTypes]]]
        Data which sums in quadrature. Providing a dictionary of lightkurve
        error products (ErrorCube, ErrorFrame, ErrorSeries) is recommended.
        However, a singular array-like or a dictionary of array-like data may
        be given and will be converted into the corresponding lkErrorType based
        on the shape of the data.

    Returns
    -------
    DataSet
        A dict-like object containing related data and error products which
        may be manipulated and analyzed simultaneously.
    """

    _user_kwargs = None

    def __init__(
        self,
        data: Union[
            Iterable, LkDataTypes, DataProducts, Dict[str, Union[Iterable, LkDataTypes]]
        ] = None,
        bools: Union[
            Iterable, LkBoolTypes, BoolProducts, Dict[str, Union[Iterable, LkBoolTypes]]
        ] = None,
        bitwise: Union[
            Iterable,
            LkBitwiseTypes,
            BitwiseProducts,
            Dict[str, Union[Iterable, LkBitwiseTypes]],
        ] = None,
        index: pd.MultiIndex = None,
        time_indices: Dict[str, Iterable] = None,
        **kwargs,
    ):
        self._user_kwargs = []
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k not in ("ntime", "nrow", "ncol", "columns"):
                self._user_kwargs.append(k)

        parsed_index = DataCube.parse_index(index, time_indices)
        if not parsed_index.empty:
            index = parsed_index

        self._data_types = {}
        if data is not None:
            if isinstance(data, DataProducts):
                self.data = data
            else:
                self.data = DataProducts(data, index, **kwargs)
            self._data_types.update(self.data._data_types)
        else:
            self.data = DataProducts(index=index, **kwargs)

        if not hasattr(self.data, "index"):
            self.data.ntime = 0
            self.data.index = parsed_index

        for val in self.data.values():
            val.index = self.index

    def __len__(self):
        return len(self.data)

    @property
    def cubes(self) -> dict:
        """Retrieve all DataCube and ErrorCube objects.

        Returns
        -------
        dict
            A dictionary containing all DataCube and ErrorCube objects from the
            Batch object. The keys are the original keys from the data and
            error dictionaries.
        """
        cubes = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataCube" in str(value)
        }

        return cubes

    @property
    def frames(self) -> dict:
        """Retrieve all DataFrame and ErrorFrame objects.

        Returns
        -------
        dict
            A dictionary containing all DataFrame and ErrorFrame objects from
            the Batch object. The keys are the original keys from the data and
            error dictionaries.
        """
        frames = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataFrame" in str(value)
        }
        return frames

    @property
    def series(self) -> dict:
        """Retrieve all DataSeries and ErrorSeries objects.

        Returns
        -------
        dict
            A dictionary containing all DataSeries and ErrorSeries objects from
            the Batch object. The keys are the original keys from the data and
            error dictionaries.
        """
        series = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataSeries" in str(value)
        }
        return series

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
        for val in newbatch.data.values():
            val.index = indices.index
        for val in newbatch.error.values():
            val.index = indices.index

        return newbatch

    def _batch_wrapper(self, func):
        def new_func(*args, **kwargs):
            newdata = dict(deepcopy(self.data))
            for key, val in newdata.items():
                obj_func = getattr(val, func)
                newdata[key] = obj_func(*args, **kwargs)

            return self._build_instance(newdata)

        return new_func

    def downsample(self, nframes: int = 5, level: Union[str, int] = -1):
        """Downsample all contained data and error products."""
        downsample = self._batch_wrapper("downsample")
        return downsample(nframes, level)

    def droplevel(self, level, axis=0):
        """Drop a given level from the index."""
        droplevel = self._batch_wrapper("droplevel")
        return droplevel(level, axis)

    @singledispatchmethod
    def __getitem__(self, key):
        pass

    @__getitem__.register
    def _(self, key: str):
        if key in self.data:
            return self.data[key]
        else:
            raise ValueError("Unrecognized key")

    @__getitem__.register(int)
    @__getitem__.register(list)
    @__getitem__.register(np.ndarray)
    def _(self, key):
        new_data = {}
        for data_key, data in self.data.items():
            new_data[data_key] = data[key]
        new_data = DataProducts(new_data, self.data.index[np.atleast_1d(key)])
        return self._build_instance(new_data)

    @__getitem__.register(slice)
    def _(self, key):
        new_data = {}
        for data_key, data in self.data.items():
            new_data[data_key] = data[key]
        new_data = DataProducts(new_data, self.data.index[key])
        return self._build_instance(new_data)

    @__getitem__.register
    def _(self, key: tuple):
        time_key = key[0]
        if len(key) not in [1, 2, 3]:
            raise KeyError(f"Cannot parse key with {len(key)} elements.")

        new_data = {
            data_key: data[key]
            for data_key, data in self.data.items()
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
                for data_key, data in self.data.items()
                if isinstance(data, (DataSeries, DataFrame))
            }
        )

        if len(new_data) > 0:
            new_index = list(new_data.values())[0].index
        else:
            new_index = pd.MultiIndex.from_arrays([[]], names=["time_index"])

        new_kwargs = self.kwargs.copy()
        new_kwargs["index"] = new_index
        new_kwargs["columns"] = new_columns

        return self._build_instance(new_data, **new_kwargs)

    @property
    def user_kwargs(self):
        """Keywords passed by the user"""
        return {key: getattr(self, key, None) for key in self._user_kwargs}

    def _attr_override(self, attr, val):
        setattr(self.data, attr, val)

    @property
    def ntime(self):
        data_val = getattr(self.data, "ntime", None)
        if data_val:
            return data_val
        else:
            return 0

    @ntime.setter
    def ntime(self, val):
        self._attr_override("ntime", val)
        if self.index.empty:
            new_index = DataCube.parse_index(ntime=val)
            self._attr_override("index", new_index)

    @property
    def index(self):
        data_val = getattr(self.data, "index", None)
        if data_val is not None:
            return data_val

        else:
            return None

    @index.setter
    def index(self, val):
        self._attr_override("index", val)

    def _build_instance(self, newdata, **kwargs):
        all_kwargs = self.user_kwargs.copy()
        all_kwargs.update(**kwargs)
        return self.__class__(newdata, **all_kwargs)

    def __repr__(self):
        if hasattr(self, "data"):
            if hasattr(self, "error"):
                msg = f"""
                Data Products:
                {self.data},
                Error Products:
                {self.error},
                Properties:
                {list(self.kwargs.keys())}
                """
            else:
                msg = f"""
                Data Products:
                {self.data}
                Properties:
                {list(self.kwargs.keys())}
                """
        elif hasattr(self, "error"):
            msg = f"""
            Error Products:
            {self.error}
            Properties:
            {list(self.kwargs.keys())}
            """
        else:
            msg = f"Empty Batch. Properties:\n{list(self.kwargs.keys())}"
        msg = dedent(msg)
        return msg
