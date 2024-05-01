import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["StatsMixin", "MathMixin", "ErrorStatsMixin", "PlotMixins"]

_AGG_ERROR_FUNCS = {
    "agg_mean": lambda x: (np.sum(x**2) ** 0.5) / len(x),
    "agg_median": lambda x: (np.sum(x**2) ** 0.5) / len(x),
    "agg_std": lambda x: np.median(x) / (np.sqrt(2 * len(x))),
    "agg_sum": lambda x: np.sum(x**2) ** 0.5,
    "agg_count": lambda x: np.isfinite(x).sum(),
}

_AGG_FUNCS = {
    "agg_mean": lambda x: np.mean(x),
    "agg_median": lambda x: np.median(x),
    "agg_std": lambda x: np.std(x),
    "agg_sum": lambda x: np.sum(x),
    "agg_count": lambda x: np.isfinite(x).sum(),
}


# Which methods should we port in from DataFrame, and update with our post_processing
STATS_METHOD_NAMES = [
    "mean",
    "median",
    "sum",
    "std",
    "var",
    "min",
    "max",
    "prod",
    "sem",
    "skew",
    "kurt",
]


CUM_METHOD_NAMES = ["cumsum", "cummin", "cummax", "cumprod"]


class StatsMixin:
    """Defines a mixin class which will let us postprocess all our pandas stats"""

    pass


def _create_stats_method(method_name):
    def _method(self, *args, **kwargs):
        pandas_method = getattr(super(pd.DataFrame, self), method_name)
        axis = kwargs.get("axis", 0)
        return self.stats_post_process(pandas_method(*args, **kwargs), axis=axis)

    return _method


def _create_cum_method(method_name):
    def _method(self, *args, **kwargs):
        pandas_method = getattr(super(pd.DataFrame, self), method_name)
        new = pandas_method(*args, **kwargs)
        return self._build_instance(new.to_numpy())

    return _method


for method_name in STATS_METHOD_NAMES:
    setattr(StatsMixin, method_name, _create_stats_method(method_name))
for method_name in CUM_METHOD_NAMES:
    setattr(StatsMixin, method_name, _create_cum_method(method_name))


class ErrorStatsMixin:
    def _sum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "sum")(axis=axis).to_numpy()

    def _mean(self, axis=0):
        return getattr(super(pd.DataFrame, self), "mean")(axis=axis).to_numpy()

    def _median(self, axis=0):
        return getattr(super(pd.DataFrame, self), "median")(axis=axis).to_numpy()

    def _cumsum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "cumsum")(axis=axis).to_numpy()

    def sum(self, axis=0):
        """Returns the standard error"""
        if axis in [0, "time"]:
            return (self**2)._sum(axis=axis).reshape(self.nrow, self.ncol) ** 0.5
        else:
            return (self**2)._sum(axis=axis) ** 0.5

    def std(self, axis=0):
        if axis in [0, "time"]:
            n = self.ntime
            return (self._median(axis=axis) / (np.sqrt(2 * n))).reshape(
                self.nrow, self.ncol
            )
        else:
            n = self.npixel
            return self._median(axis=axis) / (np.sqrt(2 * n))

    def mean(self, axis=0):
        if axis in [0, "time"]:
            n = self.ntime
        else:
            n = self.npixel
        return self.sum(axis=axis) / n

    def median(self, axis=0):
        return self.mean(axis=axis)

    def cumsum(self, axis=0):
        return (self**2)._cumsum(axis=axis) ** 0.5


class MathMixin:
    def _process_val(self, val):
        if isinstance(val, (np.ndarray, int, float)):
            return val
        elif isinstance(val, (pd.DataFrame)):
            return val.to_numpy()
        else:
            raise TypeError(f"Can not perform math operations with type {type(val)}.")

    def __add__(self, val):
        return self._build_instance(self.to_numpy() + self._process_val(val))

    def __sub__(self, val):
        return self.__add__(val)

    def __mul__(self, val):
        return self._build_instance(self.to_numpy() * self._process_val(val))

    def __pow__(self, val):
        return self._build_instance(self.to_numpy() ** self._process_val(val))

    def __mod__(self, val):
        return self._build_instance(self.to_numpy() % self._process_val(val))


class PlotMixins:
    def plot(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if isinstance(self, pd.DataFrame):
            if hasattr(self, "nrow"):
                # Assume 2D
                data = self.get_frame(kwargs.pop("frame", 0))
                ax.imshow(data, **kwargs)
            else:
                data = self.to_numpy()
                ax.plot(data, **kwargs)
        return ax
