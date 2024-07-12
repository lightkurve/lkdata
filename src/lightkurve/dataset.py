from typing import Dict, List, Union
import numpy as np


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

    _data = []
    _error = []

    def __init__(
        self,
        data: Union[Dict, List, np.ndarray] = None,
        error: Union[Dict, List, np.ndarray] = None,
        **kwargs,
    ):
        self.data = data
        self.error = error
        self.kwargs = kwargs

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, obj):
        if isinstance(obj, dict):
            for key in obj.keys():
                self.__setattr__(key, obj[key])
        else:
            self._data = np.array(obj)

    @staticmethod
    def applyfunc(obj, func, *args, **kwargs):
        func = getattr(obj, func)
        return func(*args, **kwargs)

    @classmethod
    def _build_fc_instance(cls, newdata, newerror, **kwargs):
        return cls(newdata, newerror, **kwargs)

    def __getattr__(self, attr, *args, **kwargs):
        if "_repr_" in attr:
            pass
        else:
            data_attr = getattr(self.data, attr)
            error_attr = getattr(self.error, attr)
            if callable(data_attr):

                def func(*args, **kwargs):
                    data_ret = data_attr(*args, **kwargs)
                    error_ret = error_attr(*args, **kwargs)
                    if isinstance(data_ret, self.data.__class__):
                        new = self._build_fc_instance(
                            data_ret.to_array(), error_ret.to_array(), **data_ret.meta
                        )
                        return new
                    else:
                        return (data_attr(*args, **kwargs), error_attr(*args, **kwargs))

                return func

            elif hasattr(data_attr, "__iter__"):
                if all(data_attr == error_attr):
                    return data_attr
                else:
                    return (data_attr, error_attr)
            elif data_attr == error_attr:
                return data_attr
            return (data_attr, error_attr)

    def __repr__(self):
        return f"{self.data.__repr__()}, {self.error.__repr__()}"

    def _repr_html_(self):
        return self.data._repr_html_().replace(
            self.data.__repr__(), self.data.__repr__() + ", " + self.error.__repr__()
        )
