from typing import Dict, List, Union
import numpy as np
from .datacube import DataCube, ErrorCube


class DataSet:
    """A class to hold a collection of related Data and Error data products.

    Returns
    -------
    DataSet
        This class collects data and error products (cubes, frames, and/or series)
        as well as relevant metadata. Data products of the same type must have the
        same axes, i.e. time and pixel positions. Data aggregation methods
        managed by this DataSet are applied to all contained data products,
        i.e. down sampling. Data analysis should be performed on individual
        data products, i.e. flux summation.
    """

    _data = {}
    _error = {}

    def __init__(
        self,
        data: Union[Dict, List, np.ndarray] = None,
        error: Union[Dict, List, np.ndarray] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        if data is None and error is None:
            raise ValueError("`data` or `error` must be given.")
        if data is not None:
            self.data = data
        if error is not None:
            self.error = error

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._data.keys():
                return self._data[key]
            elif key in self._error.keys():
                return self._error[key]
            else:
                raise ValueError("Unrecognized key")
        elif isinstance(key, slice):
            new_data = {}
            new_error = {}
            new_kwargs = self.kwargs.copy()
            d0 = 0
            for data_key, data in self._data.items():
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
            for error_key, error in self._error.items():
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

    @property
    def data(self):
        if len(self._data) == 1:
            return list(self._data.values())[0]
        else:
            return self._data

    @data.setter
    def data(self, obj):
        if isinstance(obj, dict):
            for key in obj.keys():
                self._data[key] = self._build_datacube(obj[key])
        elif np.array(obj).ndim == 3:
            self._data["data"] = self._build_datacube(np.array(obj))
        else:
            raise TypeError("Unrecognized input for data.")

    @property
    def error(self):
        if len(self._error) == 1:
            return list(self._error.values())[0]
        else:
            return self._error

    @error.setter
    def error(self, obj):
        if isinstance(obj, dict):
            for key in obj.keys():
                self._error[key] = self._build_errorcube(obj[key])
        elif np.array(obj).ndim == 3:
            self._error["error"] = self._build_errorcube(np.array(obj))
        else:
            raise TypeError("Unrecognized input for error.")

    def _build_datacube(self, data):
        dc = DataCube(data, **self.kwargs)
        self._check_attrs(dc)
        return dc

    def _build_errorcube(self, error):
        err_dc = ErrorCube(error, **self.kwargs)
        self._check_attrs(err_dc)
        return err_dc

    def _check_attrs(self, obj):
        for attr in ["ntime", "nrow", "ncol"]:
            if hasattr(self, attr):
                if getattr(self, attr) != getattr(obj, attr):
                    raise ValueError(
                        f"""
                        Dataset value for {attr} != given data {attr}
                        {getattr(self, attr)} != {getattr(obj, attr)}
                        """
                    )
            else:
                setattr(self, attr, getattr(obj, attr, None))
        for attr in ["index", "columns"]:
            if hasattr(self, attr):
                if getattr(self, attr).shape != getattr(obj, attr).shape:
                    raise ValueError(
                        f"""
                        Dataset shape for {attr} does not match given data {attr}.
                        {getattr(self, attr).shape} != {getattr(obj, attr).shape}
                        """
                    )

    @staticmethod
    def applyfunc(obj, funcname, *args, **kwargs):
        func = getattr(obj, funcname)
        return func(*args, **kwargs)

    @classmethod
    def _build_ds_instance(cls, newdata, newerror, **kwargs):
        return cls(newdata, newerror, **kwargs)

    def __repr__(self):
        return f"{self._data}, {self._error}"
