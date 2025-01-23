"""Mixin methods and classes for lightkurve data objects"""

import re
from copy import deepcopy
from typing import Union
from .uncertainty import NDUncertainty, Uncertainty

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

__all__ = ["StatsMixin", "MathMixin", "ErrorStatsMixin"]

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
    "sem",
    "skew",
    "kurt",
]


CUM_METHOD_NAMES = ["cumsum", "cummin", "cummax", "cumprod"]


class MathMixin:
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

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if not hasattr(value, "uncertainty_type"):
            value = Uncertainty(value)
        self._uncertainty = value

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
        if not (isinstance(self, pd.DataFrame) or isinstance(self, pd.Series)):
            raise TypeError(f"Unsupported type {type(self)}")
        kwargs = {"index": self.index}
        if isinstance(self, pd.DataFrame):
            kwargs["columns"] = self.columns
        if hasattr(self, "nrow") and hasattr(self, "ncol"):
            kwargs["nrow"] = self.nrow
            kwargs["ncol"] = self.ncol
        return kwargs

    def _arithmetic(
        self,
        operation,
        operand,
        propagate_uncertainties=True,
        uncertainty_correlation=0,
        axis=None,
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
            return operation(self.to_numpy(), self._process_math_val(operand), **kwargs)
        else:
            return operation(self.to_numpy(), **kwargs)

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
        if (
            operand is not None
            and operand.uncertainty is not None
            and not isinstance(operand.uncertainty, NDUncertainty)
        ):
            raise TypeError(
                "Uncertainty propagation is only defined for "
                "subclasses of NDUncertainty."
            )

        # Now do the uncertainty propagation
        # TODO: There is no enforced requirement that actually forbids the
        # uncertainty to have negative entries but with correlation the
        # sign of the uncertainty DOES matter.
        if self.uncertainty is None and (
            not hasattr(operand, "uncertainty") or operand.uncertainty is None
        ):
            # Neither has uncertainties so the result should have none.
            return None
        elif self.uncertainty is None:
            # Create a temporary uncertainty to allow uncertainty propagation
            # to yield the correct results. (issue #4152)
            self.uncertainty = operand.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(
                operation, operand, result, correlation
            )
            # Delete the temporary uncertainty again.
            self.uncertainty = None
            return result_uncert

        elif operand is not None and operand.uncertainty is None:
            # As with self.uncertainty is None but the other way around.
            operand.uncertainty = self.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(
                operation, operand, result, correlation
            )
            operand.uncertainty = None
            return result_uncert

        else:
            # Both have uncertainties so just propagate.

            # only supply the axis kwarg if one has been specified for a collapsing operation
            axis_kwarg = dict(axis=kwds["axis"]) if "axis" in kwds else dict()
            return self.uncertainty.propagate(
                operation, operand, result, correlation, **axis_kwarg
            )

    def __add__(self, other):
        result = self._prepare_then_do_arithmetic(np.add, other)
        return result

    def __sub__(self, other):
        result = self._prepare_then_do_arithmetic(np.subtract, other)
        return result

    def __mul__(self, other):
        result = self._prepare_then_do_arithmetic(np.multiply, other)
        return result

    def __truediv__(self, other):
        result = self._prepare_then_do_arithmetic(np.true_divide, other)
        return result

    def __pow__(self, val):
        result = self._prepare_then_do_arithmetic(np.power, val)
        return result

    def __mod__(self, val):
        result = self._prepare_then_do_arithmetic(np.mod, val)
        return result

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


class StatsMixin:
    """Defines a mixin class which will let us postprocess all our pandas stats"""

    _stats_type = "data"

    def gb_agg(self, *args, **kwargs):
        if kwargs.pop("T", False):
            data = self.T
        else:
            data = self
        gb = data.groupby(*args, **kwargs)
        return gb.agg("sum")

    def _set_stats_methods(self):
        for method_name in STATS_METHOD_NAMES:
            setattr(self, method_name, self._create_stats_method(method_name))
        for method_name in CUM_METHOD_NAMES:
            setattr(self, method_name, self._create_cum_method(method_name))

    def _create_stats_method(self, method_name):
        def _method(*args, **kwargs):
            axis = kwargs.pop("axis", 0)
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

    def _create_cum_method(self, method_name):
        def _method(*args, **kwargs):
            pandas_method = getattr(super(self._pd_class, self), method_name)
            new = pandas_method(*args, **kwargs)
            return self._build_instance(new.to_numpy())

        return _method


class ErrMathMixin(MathMixin):
    """Math mixins for lightkurve error objects."""

    def __add__(self, val):
        if isinstance(val, self.__class__()):
            val = abs(val)
            return self._build_instance(
                (self.to_numpy() ** 2 + val.to_numpy() ** 2) ** 0.5,
                **self._get_math_kwargs(),
            )
        else:
            return super().__add__(val)

    def __sub__(self, val):
        return self.__add__(-val)


class BoolMathMixin(MathMixin):
    """Math mixins for lightkurve bool objects.

    All operators should simply return the "logical or" for each element.
    """

    def __add__(self, val):
        if isinstance(val, self.__class__()):
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


class BitwiseMathMixin:
    """Math mixins for lightkurve bool objects."""

    def __add__(self, val):
        if isinstance(val, self.__class__()):
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

    def __div__(self, val):
        return self.__add__(val)


class ErrorStatsMixin:
    """Statistics mixins for error products"""

    _stats_type = "error"

    def gb_agg(self, *args, **kwargs):
        if kwargs.pop("T", False):
            data = self.T
        else:
            data = self
        gb = (data**2).groupby(*args, **kwargs)
        new = gb.agg("sum")
        return new**0.5

    def _sum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "sum")(axis=axis)

    def _mean(self, axis=0):
        return getattr(super(pd.DataFrame, self), "mean")(axis=axis)

    def _median(self, axis=0):
        return getattr(super(pd.DataFrame, self), "median")(axis=axis)

    def _cumsum(self, axis=0):
        return getattr(super(pd.DataFrame, self), "cumsum")(axis=axis)

    def new_sum(self, axis=0):
        """Returns the standard error"""
        return self.stats_post_process((self**2)._sum(axis=axis) ** 0.5, axis=axis)

    def new_std(self, axis=0):
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

    def new_mean(self, axis=0):
        if axis in [0, "time"]:
            n = self.ntime
        else:
            n = self.nseries
        return self.stats_post_process(self.sum(axis=axis) / n, axis=axis)

    def new_median(self, axis=0):
        return self.stats_post_process(self.mean(axis=axis), axis=axis)

    def new_cumsum(self, axis=0):
        return self.stats_post_process((self**2)._cumsum(axis=axis) ** 0.5, axis=axis)

    def _set_errstats_methods(self):
        for method in (
            "sum",
            "std",
            "mean",
            "median",
            "cumsum",
        ):
            setattr(self, method, getattr(self, "new_" + method))


class BoolStatsMixin:
    _stats_type = "bool"

    def gb_agg(self, *args, **kwargs):
        if kwargs.pop("T", False):
            data = self.T
        else:
            data = self
        gb = data.groupby(*args, **kwargs)
        new = gb.agg(np.logical_or.reduce)
        return new


class BitwiseMixin(MathMixin):
    """
    Mixin class that provides functionality for handling and displaying bitwise data.
    """

    _stats_type = "bitwise"
    _code_dict = None
    _values_display = None

    def gb_agg(self, *args, **kwargs):
        if kwargs.pop("T", False):
            data = self.T
        else:
            data = self

        # breakdown separates an integer flag into bitwise codes
        gb = (data.map(self.breakdown)).groupby(*args, **kwargs)
        # first sum combines lists, second sum adds unique codes together
        new = gb.agg(lambda x: np.sum(np.unique(np.sum(x)), dtype=int))
        return new

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
            raise AttributeError(f"Display must be one of {allowed}.")
        self._values_display = value.lower()
        self.styler = self.stylize_frame(self)

    @property
    def codes(self):
        """Return the code dictionary used in this Bitwise product."""
        return self._code_dict

    @codes.setter
    def codes(self, codes_dict):
        self._code_dict = codes_dict

    def breakdown(self, val):
        """
        Breaks down an integer into its constituent powers of 2.
        """
        codes = []
        asbin = bin(val)
        for pos, b in enumerate(asbin[:1:-1]):
            if int(b):
                codes.append(2 ** (pos))
        return codes

    def __add__(self, val):
        if isinstance(val, self.__class__):
            new_data = val.map(val.breakdown)
            orig_data = self.map(val.breakdown)
            updated_data = new_data.to_numpy() + orig_data.to_numpy()
            updated_data = updated_data.map(np.unique)
            return self._build_instance(
                updated_data,
                **self._get_math_kwargs(),
            )
        elif isinstance(val, int):
            orig_data = self.map(val.breakdown)
            updated_data = self.map(lambda x: x.append(val).to_numpy())
            updated_data = updated_data.map(np.unique)
            return self._build_instance(
                updated_data,
                **self._get_math_kwargs(),
            )
        else:
            raise TypeError(f"Adding value of type {type(val)} is not supported.")

    def __sub__(self, val):
        if isinstance(val, self.__class__):
            new_data = val.map(bin)
            orig_data = self.map(bin)
            updated_data = new_data.to_numpy() - orig_data.to_numpy()
            updated_data = updated_data.map(np.unique)
            return self._build_instance(
                updated_data,
                **self._get_math_kwargs(),
            )
        elif isinstance(val, int):
            orig_data = self.map(val.breakdown)
            updated_data = self.map(lambda x: x.append(val).to_numpy())
            updated_data = updated_data.map(np.unique)
            return self._build_instance(
                updated_data,
                **self._get_math_kwargs(),
            )
        else:
            raise TypeError(f"Adding value of type {type(val)} is not supported.")

    def parse_code(self, val):
        """
        Parse a bitwise integer value into a dictionary of corresponding codes.
        """
        codes = self.breakdown(val)
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
            out = out.format(
                lambda x: str(self.breakdown(x)).replace("[", "{").replace("]", "}")
            )
        elif self._values_display == "detailed":
            out = out.format(self.parse_code)

        out = out.set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": "caption-side: bottom; font-size:1em; font-weight: bold;",
                },
                {"selector": "th", "props": "text-align: center;"},
                {
                    "selector": "td",
                    "props": "height: 30px; text-align: center;",
                },
                {"selector": ":hover", "props": ""},
            ]
        )
        return out


class AggMixin:
    """ "Mixin class for data aggregation methods"""

    def _set_precision(self, func):
        """np.array wrapper to strictly enforce precision."""

        def wrap(*args, **kwargs):
            arr = func(*args, **kwargs)
            npdtype = type(arr[0])
            precision = np.finfo(npdtype).precision
            return arr.round(precision)

        return wrap

    def get_bins(self, index, nframes, right=False):
        round_arr = self._set_precision(np.array)
        # Find the average spacing of the index
        dt = np.median(np.diff(index)) * nframes
        nbins = int(np.ceil((index.max() - index.min()) / dt) + 1)
        # Calculate what bin edges result in this spacing
        bins = np.arange(index.min(), index.min() + nbins * dt, dt)
        bins = round_arr(bins)

        bin_edges = pd.cut(np.sort(index), bins, right=right)
        return bin_edges

    def downsample(self, nframes: int = 5, level: Union[int, str] = -1):
        round_arr = self._set_precision(np.array)
        # Find the index to downsample on
        index = self.index.get_level_values(level=level)
        try:
            index = round_arr(index)
        except ValueError:
            pass

        # groupby these bin edges
        bin_edges_left = self.get_bins(index, nframes, right=False)
        gb = self.groupby(bin_edges_left, observed=False)

        # We only accept cases where the number of points in a bin is the same
        # as the number of frames we downsample to
        if hasattr(self, "columns"):
            count = gb[int(self.columns.get_level_values(0)[0])].count()
            bin_mask = np.asarray(count == nframes)[:, 0]
        else:
            # for DataSeries
            count = gb.count()
            bin_mask = np.asarray(count == nframes)

        # # Downsampling aggregation depends on data type. See relevant mixin for details.
        # new = self.gb_agg(bin_edges_left, observed=False)[bin_mask]
        new = gb.agg("sum")[bin_mask]
        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            error = self.uncertainty.array.reshape(self.shape)
            error = pd.DataFrame(error**2)
            error = error.groupby(bin_edges_left, observed=False)
            error = error.agg("sum")[bin_mask].to_numpy()
            error = error**0.5

        # We have to create a new index. We take the mean of each bin.
        new_index_left = self.index.to_frame().groupby(bin_edges_left, observed=False)

        if "indices" in self.index.names:

            def repack(vals):
                # If previously downsampled, "indices" will contain strings that
                # look like lists. This combines those "lists".
                allvals = [int(num) for v in vals for num in re.findall(r"\d+", v)]
                return str(allvals)

            cadences = new_index_left["indices"].apply(repack).values
            cadences = cadences[bin_mask]
            new_index_left = (
                self.index.to_frame()
                .drop(["indices"], axis=1)
                .groupby(bin_edges_left, observed=False)
            )
            self.index = self.index.droplevel("indices")
        elif "time_index" in self.index.names:
            cadences = (
                new_index_left["time_index"]
                .apply(lambda val: str(np.unique(val)))
                .values
            )
            cadences = cadences[bin_mask]
        # if the old index was time based, use the mean of the bin for the new index
        new_index_left = new_index_left.mean().reset_index(drop=True)[bin_mask]
        index_names = list(self.index.names)
        if ("time_index" in new_index_left) or ("mid_index" in new_index_left):
            new_index_left["indices"] = cadences
            index_names.append("indices")
        new_index = new_index_left.set_index(index_names).index
        if "time_index" in new_index.names:
            new_index.set_levels(
                new_index.get_level_values("time_index").astype(int), level="time_index"
            )
            new_index = new_index.rename({"time_index": "mid_index"})

        if hasattr(self, "columns"):
            if hasattr(self, "nrow") and hasattr(self, "ncol"):
                new_obj = self._build_instance(
                    new.to_numpy(),
                    index=new_index,
                    columns=self.columns,
                    nrow=self.nrow,
                    ncol=self.ncol,
                )
            else:
                new_obj = self._build_instance(
                    new.to_numpy(),
                    index=new_index,
                    columns=self.columns,
                )
        else:
            new_obj = self._build_instance(
                new.to_numpy(),
                index=new_index,
            )

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            new_obj.uncertainty = error.reshape(new_obj.to_array().shape)

        return new_obj

    def spatial_downsample(
        self,
        factor=None,
        row_factor=None,
        col_factor=None,
        row_name=None,
        col_name=None,
    ):
        """Spatially downsamples a DataCube or a DataFrame by a given factor.

        Parameters
        ----------
        factor : int or tuple
            Factor by which to decrease the spatial resolution in each dimension.


        Returns
        -------
        new_obj : lightkurve DataCube or DataFrame
            A spatially downsampled object of the same type.


        Notes
        -----
        If `factor` is an int, this will be applied to both rows and columns.


        Examples
        --------

        """
        assert isinstance(factor, int) or isinstance(
            factor, tuple
        ), "`factor` must be an integer or a tuple of (row_factor, col_factor)"

        if isinstance(factor, int):
            row_factor = factor
            col_factor = factor
        else:
            row_factor = factor[0]
            col_factor = factor[1]
        round_array = self._set_precision(np.array)
        row_name = row_name or self.columns.names[1]
        col_name = col_name or self.columns.names[2]
        row = self.__getattribute__(row_name)
        col = self.__getattribute__(col_name)

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

        new_data = self.gb_agg(
            [bin_edges_left_row, bin_edges_left_col], T=True, observed=False
        )

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            error = self.uncertainty.array.reshape(self.shape)
            error = pd.DataFrame(error**2)
            error = error.groupby(
                [bin_edges_left_row, bin_edges_left_col], observed=False
            )
            error = error.agg("sum")[bin_mask].to_numpy()
            error = error**0.5

        new_obj = self._build_instance(
            new_data[bin_mask].T.to_numpy(),
            nrow=len(new_index[bin_mask].get_level_values(row_name).unique()),
            ncol=len(new_index[bin_mask].get_level_values(col_name).unique()),
            index=self.index,
            columns=new_index[bin_mask],
        )

        if hasattr(self, "uncertainty") and self.uncertainty.array is not None:
            new_obj.uncertainty = error.reshape(new_obj.to_array().shape)

        return new_obj

    def _expand(self, row_factor, col_factor):
        """Expands each frame of a cube maintaining relative values.

        For an element (m, n) in the original frame with a value C, the
        corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
        where rf is the row_factor and cf is the col_factor.
        """

        data = self.to_array()
        if self._stats_type == "error":
            data = data**2
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
        if self._stats_type == "error":
            return (expanded / (row_factor * col_factor)) ** 0.5
        return expanded / (row_factor * col_factor)

    def _expand_frame(self, row_factor, col_factor):
        """Expands a frame maintaining relative values.

        For an element (m, n) in the original frame with a value C, the
        corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
        where rf is the row_factor and cf is the col_factor.
        """
        data = self.to_array()
        if self._stats_type == "error":
            data = data**2
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
        if self._stats_type == "error":
            return (expanded / (row_factor * col_factor)) ** 0.5
        return expanded / (row_factor * col_factor)

    def _expand_cube_frames(self, row_factor, col_factor):
        """Expands each frame of a cube maintaining relative values.

        For an element (m, n) in the original frame with a value C, the
        corresponding (m_0...m_rf, n_0...n_cf) values will be C/(rf*cf),
        where rf is the row_factor and cf is the col_factor.
        """
        data = self.to_array()
        if self._stats_type == "error":
            data = data**2
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
        if self._stats_type == "error":
            return (expanded / (row_factor * col_factor)) ** 0.5
        return expanded / (row_factor * col_factor)

    def super_sample(self, nrows, ncols):
        pass

    def spatial_aggregate(self, nrows, ncols):
        data = self.to_array()
        row = self.__getattribute__(self.columns.names[1])
        col = self.__getattribute__(self.columns.names[2])
        assert len(data.shape) in [2, 3], "data must be a DataFrame or a DataCube"
        if len(data.shape) == 2:
            expanded_data = self._expand_frame(nrows, ncols)
            dim1 = int(expanded_data.shape[0] / nrows)
            dim2 = int(expanded_data.shape[1] / ncols)
            if self._stats_type == "error":
                expanded_data = expanded_data**2
            down_res_data = (
                expanded_data.reshape(nrows, dim1, ncols, dim2).sum(axis=1).sum(axis=2)
            )
            if self._stats_type == "error":
                down_res_data = down_res_data**0.5
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
            if self._stats_type == "error":
                expanded_data = expanded_data**2
            dim1 = int(expanded_data.shape[1] / nrows)
            dim2 = int(expanded_data.shape[2] / ncols)
            down_res_data = (
                expanded_data.reshape(dim0, nrows, dim1, ncols, dim2)
                .sum(axis=2)
                .sum(axis=3)
            )
            if self._stats_type == "error":
                down_res_data = down_res_data**0.5
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
            )
            self.nrow = old_nrow
            self.ncol = old_ncol
            return down_res_cube


class ConvenienceMixins:
    """Convenience mixins which add properties to lightkurve data objects as attributes."""

    @classmethod
    def parse_index(
        cls, index: pd.MultiIndex = None, time_indices: dict = None, ntime: int = 0
    ):
        """Parse given indices and return a single pandas MultiIndex"""
        if time_indices:
            ntime_inds = len(list(time_indices.values())[0])
            if ("time_index" not in time_indices.keys()) and (
                "mid_index" not in time_indices.keys()
            ):
                time_indices.update({"time_index": np.arange(ntime_inds)})
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

        if time_indices == {}:
            time_indices.update({"time_index": np.arange(ntime)})

        if "time_index" in time_indices:
            t0 = time_indices.pop("time_index")
            arrays = [t0, *list(time_indices.values())]
            names = ["time_index", *list(time_indices.keys())]
        elif "mid_index" in time_indices:
            t0 = time_indices.pop("mid_index")
            tfull = time_indices.pop("indices")
            arrays = [t0, tfull, *list(time_indices.values())]
            names = ["mid_index", "indices", *list(time_indices.keys())]
        else:
            arrays = [*list(time_indices.values())]
            names = [*list(time_indices.keys())]

        index = pd.MultiIndex.from_arrays(arrays, names=names)
        return index

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
            The number of rows, by default 0
        ncol : int, optional
            The number of columns, by default 0
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

        def flatten(value):
            """Flatten row and column arrays"""
            return (value * np.ones((nrow, ncol), dtype=value.dtype)).ravel()

        row_indices = row_indices or {}
        col_indices = col_indices or {}
        if (len(row_indices) > 0) and (len(col_indices) > 0) and continuous:
            nrow = len(list(row_indices.values())[0])
            ncol = len(list(col_indices.values())[0])
            for key, val in row_indices.items():
                row_indices[key] = flatten(val[:, None])
            for key, val in col_indices.items():
                col_indices[key] = flatten(val)

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
                row_indices = {"row": flatten(np.arange(nrow)[:, None])}
                col_indices = {"col": flatten(np.arange(ncol))}
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

    def sort_index(self, *args, **kwargs):
        if "inplace" in kwargs:
            inplace = kwargs["inplace"]
            kwargs["inplace"] = False
        else:
            inplace = False
        df = super(self._pd_class, self).sort_index(*args, **kwargs)
        if inplace:
            super(self._pd_class, self).__init__(df)
        else:
            return self[df.index.get_level_values(0).values]

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

    def to_array(self):
        """Method to return the data as a numpy array."""
        return self.to_numpy()

    @property
    def user_kwargs(self):
        """Keywords passed by the user"""
        return {key: getattr(self, key, None) for key in self._user_kwargs}

    def _build_instance(self, new, **kwargs):
        all_kwargs = self.user_kwargs.copy()
        all_kwargs.update(**kwargs)
        return self.__class__(new, **all_kwargs)
