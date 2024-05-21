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

    def spatial_downsample(self, factor=2):
        # Find the average spacing of the index
        dr = factor * np.median(np.diff(np.sort(np.unique(self.row))))
        dc = factor * np.median(np.diff(np.sort(np.unique(self.column))))
        flux = self
        # Calculate what bin edges result in this spacing
        bins_row = np.arange(self.row.min(), self.row.max() + 1 * dr, dr)
        bin_edges_left_row = pd.cut(np.sort(self.row), bins_row, right=False)

        bins_col = np.arange(self.column.min(), self.column.max() + 1 * dc, dc)
        bin_edges_left_col = pd.cut(self.column, bins_col, right=False)
        if self._stats_type == "error":
            gb = (self**2).T.groupby(
                [bin_edges_left_row, bin_edges_left_col], observed=False
            )
        else:
            gb = self.T.groupby(
                [bin_edges_left_row, bin_edges_left_col], observed=False
            )
        new = gb.sum()

        count = gb[int(flux.index.get_level_values(0)[0])].count()
        bin_mask = np.asarray(count == factor**2)[:, 0]

        # We have to create a new index. We'll just take the mean of each bin
        new_index_left = (
            self.columns.to_frame()
            .groupby([bin_edges_left_row, bin_edges_left_col], observed=False)
            .mean()
            .reset_index(drop=True)
        )
        new_index_right = (
            self.columns.to_frame()
            .groupby([bin_edges_left_row, bin_edges_left_col], observed=False)
            .mean()
            .reset_index(drop=True)
        )
        new_index = (
            ((new_index_left + new_index_right) / 2).set_index(self.columns.names).index
        )

        new_obj = self._build_ds_instance(
            new[bin_mask].T.to_numpy(),
            nrow=len(new_index.get_level_values("row").unique()),
            ncol=len(new_index.get_level_values("column").unique()),
            index=flux.index,
            columns=new_index[bin_mask],
        )
        if self._stats_type == "error":
            return new_obj**0.5
        else:
            return new_obj

    def _expand_frame(self, row_factor, col_factor):
        """Expands a frame maintaining relative values.

        For an element (m, n) in the original frame with a value C, the
        corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
        where rf is the row_factor and cf is the col_factor.
        """
        data = self.to_array()
        frame_size = data.shape[0] * data.shape[1]
        # Flatten to tile columns
        expanded = np.tile(data.reshape(frame_size, 1), (1, col_factor))
        # Reshape to new columns
        expanded = expanded.reshape(data.shape[0], data.shape[1] * col_factor)
        # Tile rows
        expanded = np.tile(expanded, (1, row_factor))
        # Reshape to expanded frame
        expanded = expanded.reshape(
            data.shape[0] * row_factor, data.shape[1] * col_factor
        )
        return expanded / (row_factor * col_factor)

    def _expand_cube_frames(self, row_factor, col_factor):
        """Expands each frame of a cube maintaining relative values.

        For an element (m, n) in the original frame with a value C, the
        corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
        where rf is the row_factor and cf is the col_factor.
        """
        data = self.to_array()
        frame_size = data.shape[1] * data.shape[2]
        # Flatten to tile columns
        expanded = np.tile(
            data.reshape(data.shape[0], frame_size, 1), (1, 1, col_factor)
        )
        # Reshape to new columns
        expanded = expanded.reshape(
            data.shape[0], data.shape[1], data.shape[2] * col_factor
        )
        # Tile rows
        expanded = np.tile(expanded, (1, 1, row_factor))
        # Reshape to expanded frame
        expanded = expanded.reshape(
            data.shape[0], data.shape[1] * row_factor, data.shape[2] * col_factor
        )
        return expanded / (row_factor * col_factor)

    def spatial_aggregate(self, nrows, ncols):
        data = self.to_array()
        assert len(data.shape) in [2, 3], "data must be a DataFrame or a DataCube"
        if len(data.shape) == 2:
            expanded_data = self._expand_frame(nrows, ncols)
            dim1 = int(expanded_data.shape[0] / nrows)
            dim2 = int(expanded_data.shape[1] / ncols)
            down_res_data = (
                expanded_data.reshape(nrows, dim1, ncols, dim2).sum(axis=1).sum(axis=2)
            )
            new_row_inds = (
                np.tile(self.index.values.reshape(data.shape[0], 1), (1, nrows))
                .reshape(nrows, data.shape[0])
                .mean(axis=1)
            )
            new_col_inds = (
                np.tile(self.columns.values.reshape(data.shape[1], 1), (1, ncols))
                .reshape(ncols, data.shape[1])
                .mean(axis=1)
            )

            down_res_frame = self._build_instance(
                down_res_data, index=new_row_inds, columns=new_col_inds
            )
            return down_res_frame

        else:
            dim0 = data.shape[0]
            expanded_data = self._expand_cube_frames(nrows, ncols)
            dim1 = int(expanded_data.shape[1] / nrows)
            dim2 = int(expanded_data.shape[2] / ncols)
            down_res_data = (
                expanded_data.reshape(dim0, nrows, dim1, ncols, dim2)
                .sum(axis=2)
                .sum(axis=3)
            )

            time_indices = {
                name: self.index.to_frame()[name]
                for name in self.index.names
                if name != "cadence"
            }
            new_row_inds = (
                np.tile(self.row[:: self.ncol].reshape(self.nrow, 1), (1, nrows))
                .reshape(nrows, self.nrow)
                .mean(axis=1)
            )
            new_col_inds = (
                np.tile(self.column[: self.ncol].reshape(self.ncol, 1), (1, ncols))
                .reshape(ncols, self.ncol)
                .mean(axis=1)
            )
            old_nrow = self.nrow
            old_ncol = self.ncol
            self.nrow = nrows
            self.ncol = ncols
            down_res_cube = self._build_instance(
                down_res_data,
                time_indices=time_indices,
                row_indices={"row": new_row_inds},
                col_indices={"column": new_col_inds},
            )
            self.nrow = old_nrow
            self.ncol = old_ncol
            return down_res_cube

    # def down_res_flux(self, factor, crop="centered"):
    #     """Decrease the resolution of frames by a given factor
    #     """
    #     assert isinstance(factor, int) or isinstance(factor, list), "Factor must be int or list [row factor, column factor]"
    #     flux_array = self.to_array()
    #     if isinstance(factor, int):
    #         row_factor = col_factor = factor

    #     elif isinstance(factor, list):
    #         row_factor = factor[0]
    #         col_factor = factor[1]

    #     row_remainder = flux_array.shape[1] % row_factor
    #     if row_remainder != 0:
    #         if crop == "end":
    #             flux_array = flux_array[:, :-row_remainder+1, :]
    #         elif crop == "front":
    #             flux_array = flux_array[:, row_remainder-1:, :]
    #         elif crop == "centered":
    #             if row_remainder % 2 == 0:
    #                 flux_array = flux_array[:, row_remainder/2-1:-row_remainder/2+1]
    #             else:
    #                 flux_array = flux_array[:, row_remainder/2-1:-row_remainder/2]  # can't perfectly center, dropping last row

    #     col_remainder = self.shape[2] % col_factor
    #     if col_remainder != 0:
    #         if crop == "end":
    #             flux_array = flux_array[:, :, :-col_remainder+1]
    #         elif crop == "front":
    #             flux_array = flux_array[:, :, col_remainder-1:]
    #         elif crop == "centered":
    #             if col_remainder % 2 == 0:
    #                 flux_array = flux_array[:, :, col_remainder/2-1:-col_remainder/2+1]
    #             else:
    #                 flux_array = flux_array[:, :, col_remainder/2-1:-col_remainder/2]  # can't perfectly center, dropping last column

    #     if isinstance(self, DataFrame):
    #         # to handle DataFrames
    #         new_flux_array = flux_array.reshape(flux_array.shape[1], row_factor,
    #                                             flux_array.shape[2], col_factor
    #                                             ).sum(axis=1).sum(axis=2)

    #         return new_flux_array
    #     elif isinstance(self, DataCube):
    #         # to handle DataCubes
    #         new_rows = self.row.reshape(nrows, row_factor).mean(axis=1)
    #         new_cols = self.col.reshape(ncols, col_factor).mean(axis=1)
    #         new_flux_array = flux_array.reshape(flux_array.shape[0],
    #                                             flux_array.shape[1], row_factor,
    #                                             flux_array.shape[2], col_factor
    #                                             ).sum(axis=1).sum(axis=2)
    #         new_flux = self._build_instance(
    #             new_flux_array,
    #             index = self.index.to_frame(),
    #             columns = new_cols,
    #             rows = new_rows
    #         )
    #         DataCube(new_flux_array,
    #                             self.time,
    #                             new_rows,
    #                             new_cols)
    #         pass
    #     else:
    #         pass
    #     pass


class ConvenienceMixins:
    def _include_convenience_index(self):
        INDEX_DICTS = {
            level if level is not None else "index": np.asarray(
                self.index.get_level_values(level=level)
            )
            for level in self.index.names
        }
        for key, index in INDEX_DICTS.items():
            if (key not in self._metadata) and (key != "index"):
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
            if (key not in self._metadata) and (key != "columns"):
                self._metadata.append(key)
            setattr(self, key, index)
