import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["StatsMixin", "MathMixin", "ErrorStatsMixin", "PlotMixin"]

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

    @property
    def _stats_type(self):
        return "data"


def _create_stats_method(method_name):
    def _method(self, *args, **kwargs):
        pandas_method = getattr(super(self._pd_class, self), method_name)
        axis = kwargs.get("axis", 0)
        return self.stats_post_process(pandas_method(*args, **kwargs), axis=axis)

    return _method


def _create_cum_method(method_name):
    def _method(self, *args, **kwargs):
        pandas_method = getattr(super(self._pd_class, self), method_name)
        new = pandas_method(*args, **kwargs)
        return self._build_instance(new.to_numpy())

    return _method


for method_name in STATS_METHOD_NAMES:
    setattr(StatsMixin, method_name, _create_stats_method(method_name))
for method_name in CUM_METHOD_NAMES:
    setattr(StatsMixin, method_name, _create_cum_method(method_name))


class ErrorStatsMixin:
    def _sum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "sum")(axis=axis)

    def _mean(self, axis=0):
        return getattr(super(pd.DataFrame, self), "mean")(axis=axis)

    def _median(self, axis=0):
        return getattr(super(pd.DataFrame, self), "median")(axis=axis)

    def _cumsum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "cumsum")(axis=axis)

    def sum(self, axis=0):
        """Returns the standard error"""
        return self.stats_post_process((self**2)._sum(axis=axis) ** 0.5, axis=axis)

    def std(self, axis=0):
        if axis in [0, "time"]:
            n = self.ntime
            return self.stats_post_process(
                (self._median(axis=axis) / (np.sqrt(2 * n))).reshape(
                    self.nrow, self.ncol
                ),
                axis=axis,
            )
        else:
            n = self.nseries
            return self.stats_post_process(
                self._median(axis=axis) / (np.sqrt(2 * n)), axis=axis
            )

    def mean(self, axis=0):
        if axis in [0, "time"]:
            n = self.ntime
        else:
            n = self.nseries
        return self.stats_post_process(self.sum(axis=axis) / n, axis=axis)

    def median(self, axis=0):
        return self.stats_post_process(self.mean(axis=axis), axis=axis)

    def cumsum(self, axis=0):
        return self.stats_post_process((self**2)._cumsum(axis=axis) ** 0.5, axis=axis)

    @property
    def _stats_type(self):
        return "error"


class MathMixin:
    def _process_math_val(self, val):
        if isinstance(val, (np.ndarray, float)):
            return val
        elif isinstance(val, (int, np.int64)):
            return float(val)
        elif isinstance(val, (pd.DataFrame, pd.Series)):
            return val.to_numpy()
        else:
            raise TypeError(f"Can not perform math operations with type {type(val)}.")

    def _get_math_kwargs(self):
        if isinstance(self, pd.DataFrame):
            kwargs = {"index": self.index, "columns": self.columns}
        if isinstance(self, pd.Series):
            kwargs = {"index": self.index}
        return kwargs

    def __add__(self, val):
        if self._stats_type == "error":
            if isinstance(val, self.__class__):
                return self._build_instance(
                    (self.to_numpy() ** 2 + val.to_numpy() ** 2) ** 0.5,
                    **self._get_math_kwargs(),
                )
        else:
            return self._build_instance(
                self.to_numpy() + self._process_math_val(val), **self._get_math_kwargs()
            )

    def __sub__(self, val):
        return self.__add__(-val)

    def __mul__(self, val):
        return self._build_instance(
            self.to_numpy() * self._process_math_val(val), **self._get_math_kwargs()
        )

    def __pow__(self, val):
        return self._build_instance(
            self.to_numpy() ** self._process_math_val(val), **self._get_math_kwargs()
        )

    def __mod__(self, val):
        return self._build_instance(
            self.to_numpy() % self._process_math_val(val), **self._get_math_kwargs()
        )


class PlotMixin:
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


class AggMixin:
    def downsample(self, nframes=5, level=-1):
        # Find the index to downsample on
        level = self.index.names[level] if isinstance(level, int) else level
        index = self.index.get_level_values(level=level)

        # Find the average spacing of the index
        dt = nframes * np.median(np.diff(index))
        # Calculate what bin edges result in this spacing
        bins = np.arange(index.min(), index.max() + 1 * dt, dt)
        bin_edges_left = pd.cut(np.sort(index), bins, right=False)
        # bin_edges_right = pd.cut(np.sort(index), bins, right=True)
        # groupby these bin edges

        # Downsampling is explicitly a sum
        if self._stats_type == "error":
            gb = (self**2).groupby(bin_edges_left, observed=False)
        else:
            gb = self.groupby(bin_edges_left, observed=False)

        new = gb.sum()
        # We only accept cases where the number of points in a bin is the same as the number of frames we downsample to

        count = gb[int(self.columns.get_level_values(0)[0])].count()
        bin_mask = np.asarray(count == nframes)[:, 0]

        # We have to create a new index. We'll just take the mean of each bin
        new_index_left = (
            self.index.to_frame()
            .groupby(bin_edges_left, observed=False)
            .mean()
            .reset_index(drop=True)
        )
        new_index_right = (
            self.index.to_frame()
            .groupby(bin_edges_left, observed=False)
            .mean()
            .reset_index(drop=True)
        )
        new_index = (
            ((new_index_left + new_index_right) / 2).set_index(self.index.names).index
        )

        new_obj = self._build_instance(
            new[bin_mask].to_numpy(), index=new_index[bin_mask], columns=self.columns
        )
        if self._stats_type == "error":
            return new_obj**0.5
        else:
            return new_obj


class ConvenienceMixins:
    def _include_convenience_index(self):
        INDEX_DICTS = {
            level if level is not None else "index": np.asarray(
                self.index.get_level_values(level=level)
            )
            for level in self.index.names
        }
        for key, index in INDEX_DICTS.items():
            if key not in self._metadata:
                self._metadata.append(key)
            setattr(self, key, index)

    def _include_convenience_columns(self):
        COLUMN_DICTS = {
            level if level is not None else "columns": np.asarray(
                self.columns.get_level_values(level=level)
            )
            for level in self.columns.names
        }
        for key, index in COLUMN_DICTS.items():
            if key not in self._metadata:
                self._metadata.append(key)
            setattr(self, key, index)
