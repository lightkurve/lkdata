"""Mixin methods and classes for lightkurve data objects"""

import re
from copy import deepcopy
from typing import Iterable, Union
from warnings import warn
from .uncertainty import NDUncertainty, Uncertainty
from .dtypes import BitSet

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

__all__ = ["StatsMixin", "MathMixin"]

_AGG_ERROR_FUNCS = {
    "agg_mean": lambda x: (np.sum(x**2) ** 0.5) / len(x),
    "agg_median": lambda x: (np.sum(x**2) ** 0.5) / len(x),
    "agg_std": lambda x: np.median(x) / (np.sqrt(2 * len(x))),
    "agg_sum": lambda x: np.sum(x**2) ** 0.5,
    "agg_count": lambda x: np.isfinite(x).sum(),
}

_AGG_FUNCS = {
    "agg_mean": np.mean,
    "agg_median": np.median,
    "agg_std": np.std,
    "agg_sum": np.sum,
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
    # "sem",
    # "skew",
    # "kurt",
]


CUM_METHOD_NAMES = ["cumsum", "cummin", "cummax", "cumprod"]


class IndexProcessorMixin:
    """Mixins to handle index processing"""

    def _fold_index(self, period, t0, level, label):
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
        return indices

    def _get_math_kwargs(self):
        if not (isinstance(self, pd.DataFrame) or isinstance(self, pd.Series)):
            raise TypeError(f"Unsupported type {type(self)}")
        kwargs = {"index": self.index}
        if isinstance(self, pd.DataFrame):
            kwargs["columns"] = self.columns
        if hasattr(self, "nrow") and hasattr(self, "ncol"):
            kwargs["nrow"] = self.nrow
            kwargs["ncol"] = self.ncol
        return kwargs

    def droplevel(self, level, axis=0):
        # pylint: disable:overridden-final-method
        if level in [0, "time_index", "series"]:
            raise ValueError("0-index levels cannot be dropped from Cubes.")
        if axis == 1:
            raise NotImplementedError(
                "Dropping column indices is not currently supported."
            )
        pdframe = super(self._pd_class, self).droplevel(level, axis)

        if hasattr(self, "ncol"):
            return self.from_pandas(
                pdframe,
                nrow=self.nrow,
                ncol=self.ncol,
                **self.user_kwargs,
            )
        else:
            return self.from_pandas(
                pdframe,
                **self.user_kwargs,
            )

    @staticmethod
    def parse_index(
        index: pd.MultiIndex = None, time_indices: dict = None, ntime: int = 0
    ):
        """Parse given indices and return a single pandas MultiIndex"""
        if time_indices:
            if isinstance(time_indices, dict):
                # If time_indices is given properly as a dictionary:
                if "row" in time_indices:
                    raise ValueError("Key 'row' is reserved for spatial dimensions.")
                if "col" in time_indices:
                    raise ValueError("Key 'col' is reserved for spatial dimensions.")
                ntime_inds = len(list(time_indices.values())[0])
                if ("time_index" not in time_indices.keys()) and (
                    "mid_index" not in time_indices.keys()
                ):
                    # Create a standard index which orders the data.
                    # This is particularly useful when phase-folding, etc.
                    time_indices.update({"time_index": np.arange(ntime_inds)})
            else:
                # Otherwise assume time_indices was given as an array
                time_indices = {"time_index": time_indices}
        else:
            time_indices = {}

        if isinstance(index, pd.MultiIndex):
            time_names = index.names
            time_indices.update(
                {name: index.get_level_values(name) for name in time_names}
            )

            ntime_index = len(index)
            if ("time_index" not in time_indices.keys()) and (
                "mid_index" not in time_indices.keys()
            ):
                time_indices.update({"time_index": np.arange(ntime_index)})
        elif index is not None:
            raise ValueError("'index' must be a pd.MultiIndex or None.")

        if time_indices == {}:
            time_indices.update({"time_index": np.arange(ntime)})

        if "time_index" in time_indices:
            t0 = time_indices.pop("time_index")
            arrays = [t0, *list(time_indices.values())]
            names = ["time_index", *list(time_indices.keys())]
        elif "mid_index" in time_indices:
            # For downsampled data "mid_index" is used in place of "time_index"
            t0 = time_indices.pop("mid_index")
            tfull = time_indices.pop("indices")
            arrays = [t0, tfull, *list(time_indices.values())]
            names = ["mid_index", "indices", *list(time_indices.keys())]
        else:
            arrays = [*list(time_indices.values())]
            names = [*list(time_indices.keys())]

        index = pd.MultiIndex.from_arrays(arrays, names=names)
        return index

    @staticmethod
    def parse_pos_indices(row_indices, col_indices, nrow, ncol):
        """Reshape arrays to the appropriate shape for pd.columns

        TPF data are typically stored in an intuitive 3D structure, with
        time as the 1st dimension, row (or column) as the 2nd, and the
        column (or row) as the 3rd. In using pandas as the backend for our
        data, we store time as the index of the DataFrame and need rows and
        columns to be in the DataFrame.columns.

        The standard for row and column arrays is to provide an array of
        size nrow and ncol respectively, definining the row and column
        indices.

        I.e. for a 3x4 image, a possible scenario is that
        row = [1, 2, 3]
        and col = [1, 2, 3, 4].
        In the DataFrame, this must be organized such that each column
        corresponds to one of the coordinates. So row and col must be
        flattend to
        row = [1, 1, 1, 1, 2, 2, ...,  3, 3]
        and col = [1, 2, 3, 4, 1, 2, ..., 3, 4]
        so that series[0] is [1, 1], series[2] is [1, 2], ...,
        and series[11] is [3, 4]
        """

        def process_listlike(indices, dim_self, dim_other, label, expand_method):
            """
            Check and convert the given list-like indices into compatible form
            """
            arr = np.array(indices).flatten()
            if arr.shape[0] == nrow * ncol:
                # Indices given like coordinates for each datapoint
                return arr
            elif arr.shape[0] == dim_self:
                # Indices given like markers for a grid (expected format)
                return expand_method(arr, dim_other)
            else:
                raise ValueError(
                    f"Shape of {label} does not match shape of data given."
                )

        if isinstance(row_indices, Iterable) and not isinstance(row_indices, dict):
            row = process_listlike(
                indices=row_indices,
                dim_self=nrow,
                dim_other=ncol,
                label="row",
                expand_method=np.repeat,
            )
            row_indices = {"row": row}
        row_indices = row_indices or {}

        if isinstance(row_indices, dict):
            if "time_index" in row_indices:
                raise ValueError("Key 'time_index' is reserved for time indices'")
            for key, val in row_indices.items():
                row_indices[key] = process_listlike(
                    indices=val,
                    dim_self=nrow,
                    dim_other=ncol,
                    label=key,
                    expand_method=np.repeat,
                )

        if isinstance(col_indices, Iterable) and not isinstance(col_indices, dict):
            col = process_listlike(
                indices=col_indices,
                dim_self=ncol,
                dim_other=nrow,
                label="col",
                expand_method=np.tile,
            )
            col_indices = {"col": col}

        col_indices = col_indices or {}
        if isinstance(col_indices, dict):
            if "time_index" in col_indices:
                raise ValueError("Key 'time_index' is reserved for time indices'")
            for key, val in col_indices.items():
                col_indices[key] = process_listlike(
                    indices=val,
                    dim_self=ncol,
                    dim_other=ncol,
                    label=key,
                    expand_method=np.tile,
                )

        return row_indices, col_indices

    def parse_columns(
        self,
        columns: pd.MultiIndex = None,
        row_indices: dict = None,
        col_indices: dict = None,
        nrow: int = 0,
        ncol: int = 0,
        continuous=False,
    ):
        """Parse row and column information from given information

        Parameters
        ----------
        columns : pd.MultiIndex, optional
            An existing columns instance, easiest to deal with, by default None
        row_indices : dict, optional
            A dictionary of row arrays, by default None
        col_indices : dict, optional
            A dictionary of column arrays, by default None
        nrow : int, optional
            The number of rows, by default 0. Must be defined if row_indices is
            not None.
        ncol : int, optional
            The number of columns, by default 0. Must be defined if col_indices
            is not None.
        continuous : bool, optional
            Whether the rows and columns in row and col indices should be
            interpreted as continuous.
            If not continuous, the arrays given in row and col indices should
            correspond to coordinates by pixel.
            For DataCubes, the region must be continous. For DataFrames, it is
            assumed that the region is non-contiguous, by default False.

        Returns
        -------
        pd.MultiIndex, int, int
            Returns a tuple of the parsed columns instance, the number of rows,
            and the number of columns inferred from the inputs.
        """
        if (
            (columns is None)
            and (row_indices is None)
            and (col_indices is None)
            and (nrow == 0)
            and (ncol == 0)
        ):
            return pd.MultiIndex.from_arrays([[]], names=["series"]), None, None

        row_indices, col_indices = self.parse_pos_indices(
            row_indices, col_indices, nrow, ncol
        )

        if columns is not None:
            parsed_columns = columns.to_frame().reset_index(drop=True)
            for name in [name for name in columns.names if name is not None]:
                # Attempt to parse rows and columns from columns.names
                if "row" in name:
                    row_indices.update({name: columns.get_level_values(name)})
                elif "col" in name:
                    col_indices.update({name: columns.get_level_values(name)})
        else:
            parsed_columns = pd.DataFrame()

        # If the column names weren't parseable,
        # and the given row and col indices were empty
        if (len(row_indices) == 0) or (len(col_indices) == 0):
            # Generate indices if both nrow and ncol are given
            if (nrow > 0) and (ncol > 0) and continuous:
                row_indices, col_indices = self.parse_pos_indices(
                    {"row": np.arange(nrow)}, {"col": np.arange(ncol)}, nrow, ncol
                )
            # Otherwise just return columns, rows and columns are not parseable
            else:
                return columns, nrow, ncol
        else:
            for i, val in enumerate(row_indices.values()):
                if i == 0:
                    nrow = len(np.unique(val))
                assert (
                    len(np.unique(val)) == nrow
                ), "Mismatch encountered in number of rows specified."
            for i, val in enumerate(col_indices.values()):
                if i == 0:
                    ncol = len(np.unique(val))
                assert (
                    len(np.unique(val)) == ncol
                ), "Mismatch encountered in number of columns specified."

        for key, val in row_indices.items():
            parsed_columns[key] = val
        for key, val in col_indices.items():
            parsed_columns[key] = val

        if "series" not in parsed_columns:
            parsed_columns["series"] = np.arange(nrow * ncol).ravel()

        series_col = parsed_columns.pop("series")
        parsed_columns.insert(0, "series", series_col)

        columns = parsed_columns.set_index(
            ["series", *row_indices.keys(), *col_indices.keys()]
        ).index
        self.row_names = list(row_indices.keys())
        self.col_names = list(col_indices.keys())
        return columns, nrow, ncol

    def sort_index(self, *args, **kwargs):
        init_kwds = self.user_kwargs.copy()

        inplace = kwargs.pop("inplace", False)
        pdobj = super(self._pd_class, self).sort_index(*args, **kwargs)
        if inplace:
            super(self._pd_class, self).__init__(pdobj)
        time_inds = pdobj.index.get_level_values("time_index")
        if hasattr(pdobj, "columns"):
            series_inds = pdobj.columns.get_level_values("series")
        else:
            series_inds = None

        dfarray = pdobj.to_numpy()
        if "axis" in kwargs and kwargs["axis"] in [0, "index"] or "axis" not in kwargs:
            dfarray = dfarray.reshape((self.ntime, self.nrow, self.ncol))
        if inplace:
            self.array = dfarray

        if hasattr(self, "uncertainty"):
            uncertainty_array = self.uncertainty.array
            uncertainty_array = uncertainty_array.reshape(self.shape)
            uncertainty_array = uncertainty_array[time_inds]
            if series_inds is not None:
                uncertainty_array = uncertainty_array[:, series_inds]
            uncertainty_array = uncertainty_array.reshape(dfarray.shape)
            if inplace:
                self.uncertainty.array = uncertainty_array
            else:
                init_kwds["uncertainty"] = uncertainty_array
        if inplace:
            self._include_convenience_index()
            if hasattr(self, "columns"):
                self._include_convenience_columns()
        else:
            return self.__class__.from_pandas(
                pdobj, nrow=self.nrow, ncol=self.ncol, **init_kwds
            )


class MathMixin(IndexProcessorMixin):
    """
    Mixin class to add arithmetic to an lightkurve data objects.

    Notes
    -----
    This class only aims at covering the most common cases so there are certain
    restrictions on the saved attributes::

        - ``uncertainty`` : has to be something that has a `NDUncertainty`-like
          interface for uncertainty propagation

    But there is a workaround that allows to disable handling a specific
    attribute and to simply set the results attribute to ``None`` or to
    copy the existing attribute (and neglecting the other).
    For example for uncertainties not representing an `NDUncertainty`-like
    interface you can alter the ``propagate_uncertainties`` parameter in
    :meth:`NDArithmeticMixin.add`. ``None`` means that the result will have no
    uncertainty, ``False`` means it takes the uncertainty of the first operand
    (if this does not exist from the second operand) as the result's
    uncertainty. This behavior is also explained in the docstring for the
    different arithmetic operations.
    """

    _uncertainty = None

    def __add__(self, other):
        result = self._prepare_then_do_arithmetic(np.add, other)
        return result

    def __mod__(self, val):
        result = self._prepare_then_do_arithmetic(np.mod, val)
        return result

    def __mul__(self, other):
        result = self._prepare_then_do_arithmetic(np.multiply, other)
        # Allow scalar multiplication
        if isinstance(other, (float, int, np.ndarray)) and self.uncertainty:
            result.uncertainty.array = np.multiply(result.uncertainty.array, other)
        return result

    def __pow__(self, val):
        result = self._prepare_then_do_arithmetic(np.power, val)
        return result

    def __sub__(self, other):
        result = self._prepare_then_do_arithmetic(np.subtract, other)
        return result

    def __truediv__(self, other):
        result = self._prepare_then_do_arithmetic(np.true_divide, other)
        # Allow scalar division
        if isinstance(other, (float, int, np.ndarray)) and self.uncertainty:
            result.uncertainty.array = np.true_divide(result.uncertainty.array, other)
        return result

    def _arithmetic(
        self,
        operation,
        operand,
        propagate_uncertainties=True,
        uncertainty_correlation=0,
        **kwargs,
    ):
        """
        Base method which calculates the result of the arithmetic operation.

        This method determines the result of the arithmetic operation on the
        ``data`` including their units and then forwards to other methods
        to calculate the other properties for the result (like uncertainty).

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide`.

        operand : same type (class) as self
            see :meth:`NDArithmeticMixin.add`

        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`

        uncertainty_correlation : ``Number`` or `~numpy.ndarray`, optional
            see :meth:`NDArithmeticMixin.add`

        operation_ignores_mask : bool, optional
            When True, masked values will be excluded from operations;
            otherwise the operation will be performed on all values,
            including masked ones.

        axis : int or tuple of ints, optional
            axis or axes over which to perform collapse operations like min, max, sum or mean.

        kwargs :
            Any other parameter that should be passed to the
            different :meth:`NDArithmeticMixin._arithmetic_mask` (or wcs, ...)
            methods.

        Returns
        -------
        result : ndarray
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.

        kwargs : `dict`
            The kwargs should contain all the other attributes (besides data
            and unit) needed to create a new instance for the result. Creating
            the new instance is up to the calling method, for example
            :meth:`NDArithmeticMixin.add`.

        """
        # Find the appropriate keywords for the appropriate method (not sure
        # if data and uncertainty are ever used ...)
        kwds2 = {"data": {}, "uncertainty": {}}
        for i, kwd in kwargs.items():
            splitted = i.split("_", 1)
            try:
                kwds2[splitted[0]][splitted[1]] = kwd
            except KeyError as exc:
                raise KeyError(
                    f"Unknown prefix {splitted[0]} for parameter {i}"
                ) from exc

        kwargs = {}

        result = self._arithmetic_data(operation, operand, **kwds2["data"])

        # if not hasattr(operand, "uncertainty"):
        #     propagate_uncertainties = False

        # Determine the other properties
        if propagate_uncertainties is None:
            kwargs["uncertainty"] = None
        elif not propagate_uncertainties:
            if self.uncertainty is None:
                kwargs["uncertainty"] = deepcopy(operand.uncertainty)
            else:
                kwargs["uncertainty"] = deepcopy(self.uncertainty)
        else:
            kwargs["uncertainty"] = self._arithmetic_uncertainty(
                operation,
                operand,
                result,
                uncertainty_correlation,
                **kwds2["uncertainty"],
            )

        return result, kwargs

    def _arithmetic_data(self, operation, operand, **kwargs) -> np.ndarray:
        """
        Calculate the resulting data.

        Parameters
        ----------
        operation : callable
            see `NDArithmeticMixin._arithmetic` parameter description.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        kwargs :
            Additional parameters.

        Returns
        -------
        ndarray
        """
        if operand is not None:
            return operation(self.array, self._process_math_val(operand), **kwargs)
        else:
            return operation(self, **kwargs)

    def _arithmetic_uncertainty(self, operation, operand, result, correlation, **kwds):
        """
        Calculate the resulting uncertainty.

        Parameters
        ----------
        operation : callable
            see :meth:`NDArithmeticMixin._arithmetic` parameter description.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        result : `~numpy.ndarray`
            The result of :meth:`NDArithmeticMixin._arithmetic_data`.

        correlation : number or `~numpy.ndarray`
            see :meth:`NDArithmeticMixin.add` parameter description.

        kwds :
            Additional parameters.

        Returns
        -------
        result_uncertainty : `NDUncertainty` subclass instance or None
            The resulting uncertainty already saved in the same `NDUncertainty`
            subclass that ``self`` had (or ``operand`` if self had no
            uncertainty). ``None`` only if both had no uncertainty.
        """
        # Make sure these uncertainties are NDUncertainties so this kind of
        # propagation is possible.
        if self.uncertainty is not None and not isinstance(
            self.uncertainty, NDUncertainty
        ):
            raise TypeError(
                "Uncertainty propagation is only defined for "
                "subclasses of NDUncertainty."
            )

        # Now do the uncertainty propagation
        # TODO: There is no enforced requirement that actually forbids the
        # uncertainty to have negative entries but with correlation the
        # sign of the uncertainty DOES matter.
        if not self.uncertainty and (
            not hasattr(operand, "uncertainty") or not operand.uncertainty
        ):
            # Neither has uncertainties so the result should have none.
            return None
        elif not self.uncertainty:
            # Create a temporary uncertainty to allow uncertainty propagation
            # to yield the correct results. (issue #4152)
            self.uncertainty = operand.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(
                operation, operand, result, correlation
            )
            # Delete the temporary uncertainty again.
            self.uncertainty = None
            return result_uncert

        elif operand is not None:
            if not hasattr(operand, "uncertainty"):
                # operand exists but has no uncertainty, can't propagate
                return deepcopy(self.uncertainty)
            elif not operand.uncertainty:
                # As with self.uncertainty is None but the other way around.
                operand.uncertainty = self.uncertainty.__class__(None)
                result_uncert = self.uncertainty.propagate(
                    operation, operand, result, correlation
                )
                operand.uncertainty = None
                return result_uncert

        # Both have uncertainties (or there is no operand) so just propagate.
        # only supply the axis kwarg if one has been specified for a collapsing operation
        axis_kwarg = kwds.get("axis", None)

        if axis_kwarg in [1, "series"]:
            uncertainty_copy = deepcopy(self.uncertainty)
            uncertainty_copy = uncertainty_copy.reshape(self.shape)
            return uncertainty_copy.propagate(
                operation, operand, result, correlation, axis=axis_kwarg
            )
        else:
            return self.uncertainty.propagate(
                operation, operand, result, correlation, axis=axis_kwarg
            )

    def _prepare_then_do_arithmetic(self, operation, operand):
        """Intermediate method called by public arithmetic (i.e. ``add``)
        before the processing method (``_arithmetic``) is invoked.

        .. warning::
            Do not override this method in subclasses.

        Parameters
        ----------
        operation : callable
            The operation (normally a numpy-ufunc) that represents the
            appropriate action.

        operand, operand2, kwargs :
            See for example ``add``.

        Result
        ------
        result : `~lkdata.Data`-like
        """
        # Now call the _arithmetics method to do the arithmetic.
        result, init_kwds = self._arithmetic(operation, operand)
        init_kwds.update(self._get_math_kwargs())
        # Return a new class based on the result
        return self._build_instance(result, **init_kwds)

    def _process_math_val(self, val):
        if isinstance(val, (np.ndarray, float)):
            return val
        elif isinstance(val, (int, np.int64)):
            return float(val)
        elif isinstance(val, (pd.DataFrame, pd.Series)):
            if hasattr(val, "array"):
                return val.array
            else:
                return val.to_numpy()
        else:
            raise TypeError(f"Can not perform math operations with type {type(val)}.")

    @property
    def uncertainty(self):
        """An NDData.Uncertainty object"""
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if hasattr(value, "uncertainty_type"):
            uncertainty = value
        else:
            if value is not None:
                value = self._process_math_val(value)
            if isinstance(value, (int, float)):
                value = np.ones_like(self.array) * value
            elif hasattr(value, "reshape"):
                value = value.reshape(self.array.shape)
            uncertainty = Uncertainty(value)

        uncertainty.parent_nddata = self.array
        self._uncertainty = uncertainty


class StatsMixin:
    """Defines a mixin class which will let us postprocess all our pandas stats"""

    _stats_type = "data"
    ds_agg_func = "sum"

    def _create_cum_method(self, method_name):
        def _method(*args, **kwargs):
            pandas_method = getattr(super(self._pd_class, self), method_name)
            new = pandas_method(*args, **kwargs)
            return self._build_instance(new.to_numpy())

        return _method

    def _create_stats_method(self, method_name):
        def _method(*args, **kwargs):
            axis = kwargs.pop("axis", None)
            np_method = getattr(np, method_name)
            result, init_kwds = self._arithmetic(
                np_method, operand=None, data_axis=axis, uncertainty_axis=axis, **kwargs
            )
            init_kwds.update(self._get_math_kwargs())
            result = self.stats_post_process(result, axis=axis, **init_kwds)
            return result
            # pandas_method = getattr(super(self._pd_class, self), method_name)
            # return self.stats_post_process(pandas_method(*args, **kwargs), axis=axis)

        return _method

    def _set_stats_methods(self):
        for method_name in STATS_METHOD_NAMES:
            setattr(self, method_name, self._create_stats_method(method_name))
        for method_name in CUM_METHOD_NAMES:
            setattr(self, method_name, self._create_cum_method(method_name))


class BoolMixin(IndexProcessorMixin):
    """Math mixins for lightkurve bool objects.

    All operators should simply return the "logical or" for each element.
    """

    _stats_type = "bool"
    ds_agg_func = np.logical_or.reduce

    def __add__(self, val):
        if isinstance(val, type(self)):
            return self._build_instance(
                np.logical_or(self.to_numpy(), val.to_numpy()),
                **self._get_math_kwargs(),
            )
        else:
            raise TypeError(f"Adding value type {type(val)} is not supported.")

    def __sub__(self, val):
        return self.__add__(val)

    def __mul__(self, val):
        return self.__add__(val)


class BitwiseMixin(IndexProcessorMixin):
    """
    Mixin class that provides functionality for handling bitwise data.

    Bitwise data are data which are integers in binary form. In the context of
    the Kepler and TESS Missions, flags are given as integers which, when
    converted to their binary form, indicate which flags apply to the data.
    Each flag corresponds to a bit. For an example, see Table 32 of the TESS
    Science Data Products Description Document. A value of 5 is represented in
    binary as 101, indicating that the 1st and 3rd bits are "on" corresponding
    to flags Attitude Tweak and Spacecraft is in a Coarse Point from the table.

    In aggregating bitwise data, i.e. via downsampling, we combine all flags.
    """

    _stats_type = "bitwise"
    _code_dict = None
    _values_display = None
    ds_agg_func = "sum"

    @property
    def values_display(self):
        """
        Get the current display mode for bitwise values.

        Returns
        -------
        str
            The current display mode for bitwise values. Possible values are:
            - 'bitwise': Display the raw integer values.
            - 'parsed': Display the values as sets of powers of 2.
            - 'detailed': Display the values as dictionaries mapping powers of 2 to their corresponding codes.

        Notes
        -----
        This property is used to control how bitwise values are displayed in the object's string representation
        and in any generated output (e.g., when using Jupyter notebooks).
        """
        return self._values_display

    @values_display.setter
    def values_display(self, value):
        allowed = {"bitwise", "parsed", "detailed"}
        if value.lower() not in allowed:
            warn(f"Display must be one of {allowed}, defaulting to 'bitwise'.")
            value = "bitwise"
        self._values_display = value.lower()
        self.styler = self.stylize_frame(self)

    @property
    def codes(self):
        """Return the code dictionary used in this Bitwise product."""
        return self._code_dict

    @codes.setter
    def codes(self, codes_dict):
        self._code_dict = codes_dict

    @staticmethod
    def breakdown(val):
        """
        Breaks down an integer into its constituent powers of 2.
        """
        codes = []
        asbin = bin(int(val))
        for pos, b in enumerate(asbin[:1:-1]):
            if int(b):
                codes.append(2 ** (pos))
        return codes

    @staticmethod
    @np.vectorize
    def bin_to_int(binval):
        """
        Takes a binary string and converts it to an integer
        """
        return int(binval, 2)

    def parse_code(self, val):
        """
        Parse a bitwise integer value into a dictionary of corresponding codes.
        """
        # codes = self.breakdown(val)
        codes = val
        str_codes = {code: self.codes.get(int(code), code) for code in codes}
        return str_codes

    def stylize_frame(self, df, **kwargs) -> Styler:
        """
        Overrides default to remove background gradient and to parse bitwise
        integers to a set of codes.
        """
        out = Styler(df)
        if "label" in kwargs:
            out = out.set_caption(kwargs.pop("label"))

        if (self._values_display == "parsed") or (self.codes == {}):
            out = out.format(str)
        elif self._values_display == "detailed":
            out = out.format(self.parse_code)
        else:
            out = out.format(int)

        out = out.set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": "caption-side: bottom; font-size:1em; font-weight: bold;",
                },
                {
                    "selector": "th",
                    "props": "text-align: center;",
                },
                {
                    "selector": "td",
                    "props": "height: 30px; text-align: center;",
                },
                {
                    "selector": ":hover",
                    "props": "",
                },
            ]
        )
        return out

    @staticmethod
    @np.vectorize
    def convert_set_to_int(val):
        """Convert a set to an int by summing"""
        return sum(val)

    @staticmethod
    def _set_data_type_to_int(data):
        data_arr = np.array(data)
        zeroth = data_arr.ravel()[0]
        if isinstance(zeroth, str):
            return BitwiseMixin.bin_to_int(data_arr)
        elif isinstance(zeroth, set):
            return BitwiseMixin.convert_set_to_int(data_arr)
        else:
            # if it's not a numeric type, this will raise an error
            try:
                return data_arr.astype(int)
            except (ValueError, TypeError) as e:
                raise ValueError("Unable to convert data given to integer type.") from e

    @staticmethod
    def _set_data_type_to_bitset(data):
        data_arr = np.array(data)
        convert = np.vectorize(BitSet)
        return convert(data_arr)


def _expand_frame(data, row_factor, col_factor):
    """Expands a frame maintaining relative values.

    For an element (m, n) in the original frame with a value C, the
    corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
    where rf is the row_factor and cf is the col_factor.
    """

    frame_size = data.shape[0] * data.shape[1]
    # Flatten to tile columns
    expanded = np.tile(data.reshape(frame_size, 1), (1, col_factor))
    # Reshape to new columns
    expanded = expanded.reshape(data.shape[0], data.shape[1] * col_factor)
    # Tile rows
    expanded = np.tile(expanded, (1, row_factor))
    # Reshape to expanded frame
    expanded = expanded.reshape(data.shape[0] * row_factor, data.shape[1] * col_factor)
    return expanded / (row_factor * col_factor)


def _expand_cube_frames(data, row_factor, col_factor):
    """Expands each frame of a cube maintaining relative values.

    For an element (m, n) in the original frame with a value C, the
    corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
    where rf is the row_factor and cf is the col_factor.
    """
    frame_size = data.shape[1] * data.shape[2]
    # Flatten to tile columns
    expanded = np.tile(data.reshape(data.shape[0], frame_size, 1), (1, 1, col_factor))
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


class AggMixin:
    """Mixin class for data aggregation methods like downsampling"""

    @staticmethod
    def _set_precision(func):
        """np.array wrapper to strictly enforce precision."""

        def wrap(*args, **kwargs) -> np.ndarray:
            arr = func(*args, **kwargs)
            npfinfo = np.finfo(type(arr[0]))
            if hasattr(npfinfo, "precision"):
                precision = npfinfo.precision
                return arr.round(precision)
            return arr

        return wrap

    @staticmethod
    def get_bins(index: np.ndarray, nframes: int, right=False):
        """Calculate bin edges for downsampling.

        Parameters
        ----------
        index : array-like
            The index to be binned.
        nframes : int
            Number of frames to average over.
        right : bool, optional
            Whether the intervals should be closed on the right (default: False).

        Returns
        -------
        pandas.IntervalIndex
            The bin edges for downsampling.
        """
        index.sort()
        # Find the average spacing of the index
        dt = np.median(np.diff(index)) * nframes
        nbins = int(np.ceil((index.max() - index.min()) / dt) + 1)
        # Calculate what bin edges result in this spacing
        bins = np.arange(index.min(), index.min() + nbins * dt, dt)
        round_arr = AggMixin._set_precision(np.array)
        bins = round_arr(bins)

        bin_edges = pd.cut(np.sort(index), bins, right=right)
        return bin_edges

    def downsample(self, nframes: int = 5, level: Union[int, str] = -1):
        """Downsample the data by averaging over `nframes` consecutive rows.

        Parameters
        ----------
        nframes : int, optional
            Number of frames to average over. Default is 5.
        level : Union[int, str], optional
            Index level to use for downsampling. Default is -1 (last level).

        Returns
        -------
        Same type as self
            A new object with downsampled data.

        Notes
        -----
        This method works by creating bins of `nframes` consecutive rows,
        then averaging the data within each bin. Only bins with exactly
        `nframes` rows are included in the result.

        If the object has an uncertainty attribute, it will be propagated
        by summing the squares of the uncertainties within each bin and
        then taking the square root.

        The resulting object will have a new index that represents the
        mean of the original indices within each bin. If the original
        index included a 'time_index' or 'indices' level, this information
        is preserved in the new index.
        """
        round_arr = AggMixin._set_precision(np.array)
        dfcopy = self.iloc[:]
        # Get the values of the index on which to downsample
        index = dfcopy.index.get_level_values(level=level)
        index_names = list(dfcopy.index.names)

        try:
            index = round_arr(index)
        except ValueError:
            # Can't round integers
            index = np.array(index)

        # groupby these bin edges
        bin_edges_left = AggMixin.get_bins(index, nframes, right=False)
        gb = dfcopy.groupby(bin_edges_left, observed=False)

        # We only accept cases where the number of points in a bin is the same
        # as the number of frames we downsample to
        if hasattr(dfcopy, "columns"):
            count = gb[int(dfcopy.columns.get_level_values(0)[0])].count()
            bin_mask = np.asarray(count == nframes)[:, 0]
        else:
            # for DataSeries
            count = gb.count()
            bin_mask = np.asarray(count == nframes)

        # Downsampling aggregation depends on data type.
        # See relevant mixin for details.
        # I.e. for numerical data:
        # new = gb.agg("sum")[bin_mask]
        new = gb.agg(self.ds_agg_func)[bin_mask]

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            error = self.uncertainty.array.reshape(self.shape)
            error = pd.DataFrame(error**2)
            error = error.groupby(bin_edges_left, observed=False)
            error = error.agg("sum")[bin_mask].to_numpy()
            error = error**0.5

        # We have to create a new index.
        new_index_gb = dfcopy.index.to_frame().groupby(bin_edges_left, observed=False)

        if "indices" in index_names:
            # If previously downsampled, "indices" will contain strings that
            # look like lists. This combines those "lists".
            def repack(vals):
                allvals = [int(num) for v in vals for num in re.findall(r"\d+", v)]
                return str(allvals)

            indices_in_bin = new_index_gb["indices"].apply(repack).values
            index_names.remove("indices")  # remove so groupby.mean can work
            new_index_gb = new_index_gb[index_names]

        elif "time_index" in index_names:
            indices_in_bin = (
                new_index_gb["time_index"].apply(lambda val: str(np.unique(val))).values
            )

        elif "mid_index" in index_names:
            # "mid_index" should only exist when "indices" does, but if it was
            # removed, carry on by combining the mid_index values combined
            indices_in_bin = (
                new_index_gb["mid_index"].apply(lambda val: str(np.unique(val))).values
            )

        else:
            indices_in_bin = []

        # use the mean of the bin for the new index
        new_index_left = new_index_gb.mean().reset_index(drop=True)[bin_mask]
        if len(indices_in_bin) > 0:
            new_index_left["indices"] = indices_in_bin[bin_mask]
            index_names.append("indices")
        new_index = new_index_left.set_index(index_names).index

        if "time_index" in new_index.names:
            # Ensure index is int
            t_ind_as_int = new_index.get_level_values("time_index").astype(int)
            try:
                new_index.set_levels(t_ind_as_int, level="time_index")
            except ValueError:
                # Non-linear time indices (possibly folded)
                # -> possible repeat mid_index when forced to int
                new_index.set_levels(
                    new_index.get_level_values("time_index"), level="time_index"
                )
            # Rename "time_index" to "mid_index" to reflect downsampling
            new_index = new_index.rename({"time_index": "mid_index"})

        if hasattr(self, "columns"):
            if hasattr(self, "nrow") and hasattr(self, "ncol"):
                # Cube (maybe Frame)
                new_obj = self._build_instance(
                    new.to_numpy(),
                    index=new_index,
                    columns=self.columns,
                    nrow=self.nrow,
                    ncol=self.ncol,
                )
            else:
                # Frame
                new_obj = self._build_instance(
                    new.to_numpy(),
                    index=new_index,
                    columns=self.columns,
                )
        else:
            # Series
            new_obj = self._build_instance(
                new.to_numpy(),
                index=new_index,
            )

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            new_obj.uncertainty = error.reshape(new_obj.array.shape)

        return new_obj

    def spatial_downsample(
        self, factor=None, col_factor=None, row_name=None, col_name=None, **kwargs
    ):
        """Spatially downsamples a DataCube by a given factor.

        Parameters
        ----------
        factor : int or tuple
            If a tuple is given, the first value will be used as the factor by
            which to reduce the size of the row axis, and the second as the
            column factor.
            If factor is an integer and col_factor is also given, this is the
            factor by which to decrease the spatial resolution of the row axis.
            If col_factor is not given, this is the row and column factor.
        col_factor : int, optional
            Factor by which to decrease the spatial resolution of the column
            axis. If `factor` is a tuple, this argument is ignored.
        row_name : str, optional
            Name of the axis corresponding to the row to be downsampled. By
            default the primary row axis is used.
        col_name : str, optional
            Name of the axis corresponding to the column to be downsampled. By
            default the primary column axis is used.

        Returns
        -------
        lkdata.DataCube
            A spatially downsampled object of the same type.

        TODO: This shouldn't be a mixin for lk products, it's Cube specific
        with an application to a non-timeseries DataFrame and isn't meaningful
        for Series at all.
        """
        factor = kwargs.get("row_factor", factor)
        if isinstance(factor, int):
            row_factor = factor
            if col_factor is None:
                col_factor = factor
            elif not isinstance(col_factor, int):
                raise ValueError("`col_factor` must be an integer.")
        elif isinstance(factor, tuple):
            row_factor = factor[0]
            col_factor = factor[1]
        else:
            raise (
                ValueError(
                    "`factor` must be an integer or a tuple of (row_factor, col_factor)"
                )
            )
        round_array = self._set_precision(np.array)
        row = getattr(self, row_name, self.row_names[0])
        col = getattr(self, col_name, self.col_names[0])

        if row.dtype == int and col.dtype == int:
            indexed = True
        else:
            indexed = False
            row = round_array(row)
            col = round_array(col)

        # Find the average spacing of the index
        dr = np.median(np.diff(np.sort(np.unique(row))))
        dc = np.median(np.diff(np.sort(np.unique(col))))

        # Calculate what bin edges result in this spacing
        if indexed:
            bins_row = np.arange(
                row.min(), row.max() + 1 + row_factor * dr, row_factor * dr
            )
            bins_col = np.arange(
                col.min(), col.max() + 1 + col_factor * dc, col_factor * dc
            )
        else:
            bins_row = np.arange(
                row.min(), row.max() + (row_factor - 1) * dr, row_factor * dr
            )
            bins_col = np.arange(
                col.min(), col.max() + (col_factor - 1) * dc, col_factor * dc
            )

        bin_edges_left_row = pd.cut(np.sort(row), bins_row, right=False)
        bin_edges_left_col = pd.cut(col, bins_col, right=False)

        gb = self.T.groupby([bin_edges_left_row, bin_edges_left_col], observed=False)
        count = gb[int(self.index.get_level_values(0)[0])].count()
        bin_mask = np.asarray(count == row_factor * col_factor)[:, 0]

        new_index_left = self.columns.to_frame().groupby(
            [bin_edges_left_row, bin_edges_left_col], observed=False
        )
        if indexed:
            # If the old indices weren't positional, use min index for each bin for new index
            new_index_left = new_index_left.min().reset_index(drop=True)
        else:
            # If old indices were positional, use mean of bin for new index
            new_index_left = new_index_left.mean().reset_index(drop=True)

        new_index = new_index_left.set_index(self.columns.names).index

        new_data = self.ds_agg(
            [bin_edges_left_row, bin_edges_left_col], T=True, observed=False
        )

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            error = self.uncertainty.array.reshape(self.shape)
            error = pd.DataFrame(error**2).T
            error = error.groupby(
                [bin_edges_left_row, bin_edges_left_col], observed=False
            )
            error = error.agg("sum")[bin_mask].to_numpy()
            error = error**0.5
        else:
            error = None

        new_obj = self._build_instance(
            new_data[bin_mask].T.to_numpy(),
            nrow=len(new_index[bin_mask].get_level_values(row_name).unique()),
            ncol=len(new_index[bin_mask].get_level_values(col_name).unique()),
            index=self.index,
            columns=new_index[bin_mask],
        )

        if error is not None:
            new_obj.uncertainty = error.T.reshape(new_obj.array.shape)

        return new_obj

    def super_sample(self, nrows, ncols):
        """Split pixels for super sampling"""
        pass

    def spatial_aggregate(self, nrows, ncols):
        """Similar to spatial downsample, but specify desired dimensions

        TODO: This shouldn't be a mixin for lk products, it's Cube specific
        with an application to a non-timeseries DataFrame and isn't
        meaningful for series at all.
        """
        data = self.array
        row = getattr(self, self.row_names[0])
        col = getattr(self, self.col_names[0])
        assert len(data.shape) in [2, 3], "data must be a DataFrame or a DataCube"
        # Frames
        if len(data.shape) == 2:
            # Process values
            expanded_data = self._expand_frame(data, nrows, ncols)
            dim1 = int(expanded_data.shape[0] / nrows)
            dim2 = int(expanded_data.shape[1] / ncols)

            down_res_data = (
                expanded_data.reshape(nrows, dim1, ncols, dim2).sum(axis=1).sum(axis=2)
            )

            # Process uncertainty
            if (
                hasattr(self, "uncertainty")
                & (self.uncertainty is not None)
                & (self.uncertainty.array is not None)
            ):
                # This uncertainty math is sketchy,
                # but consistent with downsample. Grain of salt.
                expanded_unc = _expand_frame(self.uncertainty.array**2, nrows, ncols)
                down_res_unc = (
                    expanded_unc.reshape(nrows, dim1, ncols, dim2)
                    .sum(axis=1)
                    .sum(axis=2)
                )
                down_res_unc = down_res_unc**0.5
            else:
                down_res_unc = None

            # Get new column values
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
                down_res_data,
                index=new_row_inds,
                columns=new_col_inds,
                uncertainty=down_res_unc,
            )
            return down_res_frame

        # Cubes
        else:
            dim0 = data.shape[0]
            expanded_data = _expand_cube_frames(data, nrows, ncols)
            dim1 = int(expanded_data.shape[1] / nrows)
            dim2 = int(expanded_data.shape[2] / ncols)
            down_res_data = (
                expanded_data.reshape(dim0, nrows, dim1, ncols, dim2)
                .sum(axis=2)
                .sum(axis=3)
            )
            if (
                hasattr(self, "uncertainty")
                & (self.uncertainty is not None)
                & (self.uncertainty.array is not None)
            ):
                # This uncertainty math is sketchy,
                # but consistent with downsample. Grain of salt.
                uncertainty = self.uncertainty.array.reshape(data.shape)
                expanded_unc = _expand_cube_frames(uncertainty**2, nrows, ncols)
                down_res_unc = (
                    expanded_unc.reshape(dim0, nrows, dim1, ncols, dim2)
                    .sum(axis=2)
                    .sum(axis=3)
                )
                down_res_unc = down_res_unc**0.5
            else:
                down_res_unc = None

            time_indices = {
                name: self.index.to_frame()[name]
                for name in self.index.names
                if name != "cadence"
            }
            new_row_inds = (
                np.tile(row[:: self.ncol].reshape(self.nrow, 1), (1, nrows))
                .reshape(nrows, self.nrow)
                .mean(axis=1)
            )
            new_col_inds = (
                np.tile(col[: self.ncol].reshape(self.ncol, 1), (1, ncols))
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
                uncertainty=down_res_unc,
            )
            self.nrow = old_nrow
            self.ncol = old_ncol
            return down_res_cube


class ConvenienceMixins:
    """Convenience mixins which add properties to lightkurve data objects as attributes."""

    def _build_instance(self, new, **kwargs):
        all_kwargs = self.user_kwargs.copy()
        all_kwargs.update(**kwargs)
        return self.__class__(new, **all_kwargs)

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

    def fold(
        self,
        period: float,
        t0: float = None,
        level: Union[int, str] = 1,
        inplace: bool = False,
        label: str = "phase",
    ):
        """Fold data on a given period and adds the folded time as an index.

        Parameters
        ----------
        period : float
            The period on which to fold the data. The user must ensure that it
            this has the same units and scale as the level.
        t0 : float, optional
            The time at which to start the first period, by default None and t0
            becomes the minimum value of the time array.
        level : Union[int, str], optional
            The index level on which to fold, by default 1, presumed to be the
            first time index that aren't cadences.
        inplace : bool, optional
            Whether to modify the object itself or return a new object, by default False
        label : str, optional
            What label to give the new time index, by default "phase"

        Returns
        -------
        Union[Cube, Frame, Series]
            Returns an object of the same type given.
        """
        indices = self._fold_index(period, t0, level, label)
        indices.set_index(label, append=True, inplace=True)
        if inplace:
            new_data_obj = self
        else:
            new_data_obj = deepcopy(self)

        setattr(new_data_obj, "index", indices.index)
        new_data_obj._metadata.append(label)
        setattr(new_data_obj, label, new_data_obj.index.get_level_values(level=label))
        return new_data_obj

    @property
    def user_kwargs(self):
        """Keywords passed by the user"""
        return {key: getattr(self, key, None) for key in self._user_kwargs}
