"""Classes and tools for creating data bundles and batches"""
from typing import Dict, List, Union
from abc import ABC
from functools import singledispatchmethod
import numpy as np
from .datacube import DataCube, ErrorCube
from .dataframe import DataFrame, ErrorFrame
from .dataseries import DataSeries, ErrorSeries


class DataProcessor(ABC):
    CLASS_CHECKS = {
        DataCube: {"ntime", "nrow", "ncol", "index", "columns"},
        ErrorCube: {"ntime", "nrow", "ncol", "index", "columns"},
        DataFrame: {"ntime", "npix", "index"},
        ErrorFrame: {"ntime", "npix", "index"},
        DataSeries: {"ntime", "index"},
        ErrorSeries: {"ntime", "index"},
    }
    _data = {}
    _error = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _check_attrs(self, data_product):
        attrs = self.CLASS_CHECKS[data_product.__class__]
        for attr in attrs.intersection({"ntime", "nrow", "ncol", "npix"}):
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

    def _build_data_product(self, data_arr: Union[List, np.ndarray], kind: str):
        data_arr = np.array(data_arr)
        data_classes = {3: DataCube, 2: DataFrame, 1: DataSeries}
        error_classes = {3: ErrorCube, 2: ErrorFrame, 1: ErrorSeries}
        classes = {"data": data_classes, "error": error_classes}

        obj_class = classes[kind].get(data_arr.ndim, None)
        if obj_class:
            data_product = obj_class(data_arr, **self.kwargs)
            self._check_attrs(data_product)
        else:
            raise TypeError("Unrecognized format given for input.")
        return data_product

    def process_input(self, data_input: Union[list, np.ndarray], kind: str) -> None:
        assert kind in ["data", "error"], "Unrecognized type"
        try:
            data_product = self._build_data_product(data_input, kind)
            return data_product
        except TypeError as err:
            raise TypeError(
                "If giving multiple data products, provide input as a dictionary."
            ) from err


class BaseBundle(dict, DataProcessor):
    """A class to hold a collection of related Data and Error data products.

    Returns
    -------
    DataBundle
        This class collects data and error products (cubes, frames, and/or series)
        as well as relevant metadata. Data products of the same type must have the
        same axes, i.e. time and pixel positions. Data aggregation methods
        managed by this DataSet are applied to all contained data products,
        i.e. down sampling. Data analysis should be performed on individual
        data products, i.e. flux summation.
    """

    _processing = None

    def __init__(
        self,
        data: Union[Dict, List, np.ndarray] = None,
        **kwargs,
    ):
        super().__init__()
        self._data_types = dict()
        data = self._unpack_data(data)
        self.update(data)

    @singledispatchmethod
    def _unpack_data(self, data):
        raise ValueError(f"Unsupported data type {type(data)}")

    @_unpack_data.register
    def _(self, data: dict):
        return data

    @_unpack_data.register
    def _(self, data: Union[list, np.ndarray]):
        return {self._processing: np.array(data)}

    def __setitem__(self, key, val):
        val = self.process_input(val, kind=self._processing)
        self._data_types[key] = type(val)
        dict.__setitem__(self, key, val)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class DataBundle(BaseBundle):
    _processing = "data"

    def __init__(
        self,
        data: Union[Dict, List, np.ndarray] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(data)


class ErrorBundle(BaseBundle):
    _processing = "error"

    def __init__(
        self,
        error: Union[Dict, List, np.ndarray] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__(error)


class Batch:
    def __init__(self, data, error, **kwargs):
        self.kwargs = kwargs
        self.data = DataBundle(data, **kwargs)
        self.error = ErrorBundle(error, **kwargs)
        self._data_types = self.data._data_types
        self._data_types.update(self.error._data_types)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def cubes(self):
        cubes = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataCube" in str(value)
        }
        cubes.update(
            {
                key: self.error[key]
                for key, value in self._data_types.items()
                if "ErrorCube" in str(value)
            }
        )
        return cubes

    @property
    def frames(self):
        frames = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataFrame" in str(value)
        }
        frames.update(
            {
                key: self.error[key]
                for key, value in self._data_types.items()
                if "ErrorFrame" in str(value)
            }
        )
        return frames

    @property
    def series(self):
        series = {
            key: self.data[key]
            for key, value in self._data_types.items()
            if "DataSeries" in str(value)
        }
        series.update(
            {
                key: self.error[key]
                for key, value in self._data_types.items()
                if "ErrorSeries" in str(value)
            }
        )
        return series

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.data:
                return self.data[key]
            elif key in self.error:
                return self.error[key]
            else:
                raise ValueError("Unrecognized key")
        elif isinstance(key, slice):
            new_data = {}
            new_error = {}
            new_kwargs = self.kwargs.copy()
            d0 = 0
            for data_key, data in self.data.items():
                if isinstance(d0, int):
                    d0 = data[key]
                    new_kwargs["row_indices"] = {
                        row_name: None for row_name in d0.row_names
                    }
                    new_kwargs["col_indices"] = {
                        col_name: None for col_name in d0.col_names
                    }
                    new_kwargs["index"] = d0.index
                    new_kwargs["columns"] = d0.columns
                new_data[data_key] = data[key].to_array()
            for error_key, error in self.error.items():
                if isinstance(d0, int):
                    d0 = error[key]
                    new_kwargs["row_indices"] = {
                        row_name: None for row_name in d0.row_names
                    }
                    new_kwargs["col_indices"] = {
                        col_name: None for col_name in d0.col_names
                    }
                    new_kwargs["index"] = d0.index
                    new_kwargs["columns"] = d0.columns
                new_error[error_key] = error[key].to_array()
            return self._build_ds_instance(new_data, new_error, **new_kwargs)

    @classmethod
    def _build_ds_instance(cls, newdata, newerror, **kwargs):
        return cls(newdata, newerror, **kwargs)

    def __repr__(self):
        return f"{self.data},\n{self.error}"
