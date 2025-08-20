"""Mixin methods and classes for lightkurve data objects"""

import re
from copy import deepcopy
from itertools import combinations
from textwrap import dedent
from typing import Iterable, Union, Tuple
from warnings import warn
from .uncertainty import NDUncertainty, Uncertainty
from .bitset import BitSet

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
    "sum",
    "std",
    "var",
    "argmin",
    "argmax",
    "min",
    "max",
    "prod",
    # "sem",
    # "skew",
    # "kurt",
]


CUM_METHOD_NAMES = ["cumsum", "cummin", "cummax", "cumprod"]


class IndexProcessorMixin:
    """A mixin class that provides methods for processing and manipulating index-related operations.

    This mixin is designed to be used with pandas-based data structures and provides
    utilities for folding time series, parsing indices, and handling various index
    operations specific to astronomical time series data.

    Methods
    -------
    droplevel(level, axis=0)
        Drop a level from the index while preserving the data structure.

    parse_index(index=None, time_indices=None, ntime=0)
        Parse given indices and return a single pandas MultiIndex.

    parse_pos_indices(row_indices, col_indices, nrow, ncol)
        Parse positional indices and reshape arrays to the appropriate shape
        for pandas columns.

    parse_columns(columns=None, row_indices=None, col_indices=None, nrow=0, ncol=0, continuous=False, nseries=0)
        Parse row and column information from given inputs.

    sort_index(*args, **kwargs)
        Sort the index of the data structure, wraps pandas sort_index methods.

    See Also
    --------
        pandas.Index : The basic object storing axis labels for all pandas objects.

    Notes
    -----
    This mixin is particularly useful for handling complex index structures in
    astronomical time series data, such as those found in Kepler and TESS observations.
    It provides methods for parsing and manipulating indices, which is crucial for
    operations like folding light curves and handling spatial information in image data.

    The methods in this mixin assume that the class it's mixed into has certain
    attributes and methods typical of pandas-based data structures, such as `index`,
    `columns`, and pandas-style indexing operations.
    """

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
        """Drop a level from the index, dropping columns this way is disabled

        Parameters
        ----------
        level : int or str
            The level to be dropped, cannot be the 0th level
        axis : {0 or 'index'}, default 0
            axis must be 0, dropping columns is not supported. Included for
            consistency with pandas.

        Returns
        -------
        self.__class__
            Returns a new instance of the calling class with the index dropped

        Raises
        ------
        ValueError
            0-level indices cannot be dropped by this method.
        NotImplementedError
            _description_

        See Also
        --------
        pandas.DataFrame.droplevel : method to drop levels from DataFrames
        pandas.Series.droplevel : method to drop levels from Series
        """
        if level in [0, "time_index", "series"]:
            raise ValueError("0-index levels cannot be dropped from lk classes.")
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
        index: pd.MultiIndex = None,
        time_indices: dict = None,
        ntime: int = 0,
        default: bool = True,
    ):
        """Parse given indices and return a single pandas MultiIndex
        Parameters
        ----------
        index : pd.MultiIndex, optional
            An existing pandas MultiIndex to be parsed. If provided, its levels
            will be incorporated into the resulting index.
        time_indices : dict or array-like, optional
            A dictionary of time-related indices or an array of time values.
            If a dictionary, keys represent index names and values are the
            corresponding arrays. Special keys 'row' and 'col' are reserved
            and will raise an error if used.
        ntime : int, optional
            The number of time points. Used to create a default time index if
            no other time information is provided. Default is 0, overwritten by
            the shape of any other given  parameter.
        Returns
        -------
        pd.MultiIndex
            A pandas MultiIndex constructed from the input parameters.

        Raises
        ------
        ValueError
            If 'row' or 'col' keys are present in time_indices.
            If 'index' is provided but is not a pd.MultiIndex.

        Notes
        -----
        - If neither 'time_index' nor 'mid_index' are present in the input,
        a default 'time_index' will be created using numpy.arange.
        - For downsampled data, 'mid_index' is used in place of 'time_index',
        and an additional 'indices' level is included, containing a string of
        all indices aggregated for the row.
        - The method prioritizes existing index information, falling back to
        provided time_indices, and finally to a default range index if necessary.
        """
        if time_indices:
            if isinstance(time_indices, dict):
                # If time_indices is given properly as a dictionary:
                if "row" in time_indices:
                    msg = dedent(
                        """\
                        Key 'row' is reserved for spatial dimensions. Rename or
                        remove the offending item from `time_indices`.
                        """
                    ).replace("\n", " ")
                    raise ValueError(msg)
                if "col" in time_indices:
                    msg = dedent(
                        """\
                        Key 'col' is reserved for spatial dimensions. Rename or
                        remove the offending item from `time_indices`.
                        """
                    ).replace("\n", " ")
                    raise ValueError(msg)
                ntime_inds = len(list(time_indices.values())[0])
                if (
                    ("time_index" not in time_indices.keys())
                    and ("mid_index" not in time_indices.keys())
                    and default
                ):
                    # Create a standard index which orders the data.
                    # This is particularly useful when phase-folding, etc.
                    time_indices.update({"time_index": np.arange(ntime_inds)})
            else:
                # Otherwise assume time_indices was given as an array
                time_indices = {"time_index": time_indices}
        else:
            time_indices = {}

        if isinstance(index, pd.Index):
            time_names = index.names
            time_indices.update(
                {name: index.get_level_values(name) for name in time_names}
            )

            ntime_index = len(index)
            if (
                ("time_index" not in time_indices.keys())
                and ("mid_index" not in time_indices.keys())
                and default
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

        # Check for and drop any duplicates
        combos = list(combinations(range(len(arrays)), 2))
        dupes = []
        for combo in combos:
            if all(arrays[combo[0]] == arrays[combo[1]]):
                dupes.append(combo[1])
        for dupe in dupes:
            del (arrays[dupe], names[dupe])

        index = pd.MultiIndex.from_arrays(arrays, names=names)
        return index

    @staticmethod
    def parse_pos_indices(row_indices, col_indices, nrow, ncol):
        """Parse and process row and column indices for data representation.

        TPF data are typically stored in an intuitive 3D structure, with
        time as the 1st dimension, row (or column) as the 2nd, and the
        column (or row) as the 3rd. In using pandas as the backend for our
        data, we store time as the index of the DataFrame and need rows and
        columns to be in the DataFrame.columns.

        This method processes the given row and column indices, ensuring they are in the
        correct format and shape for the data representation. It handles various input
        types and converts them into a standardized dictionary format.

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

        Parameters
        ----------
        row_indices : int, array-like, or dict
            The row indices. Can be an integer (starting index), an array-like object,
            or a dictionary of named row indices.
        col_indices : int, array-like, or dict
            The column indices. Can be an integer (starting index), an array-like object,
            or a dictionary of named column indices.
        nrow : int
            The number of rows in the data.
        ncol : int
            The number of columns in the data.

        Returns
        -------
        tuple of dicts
            A tuple containing two dictionaries:
            - The first dictionary contains the processed row indices.
            - The second dictionary contains the processed column indices.

        Raises
        ------
        ValueError
            If the shape of the provided indices doesn't match the shape of the data,
            or if 'time_index' is used as a key in row_indices or col_indices.

        Notes
        -----
        - If row_indices or col_indices is an integer, it's interpreted as the starting
        index, and a range is created.
        - If row_indices or col_indices is an array-like object, it's processed to ensure
        compatibility with the data shape.
        - If row_indices or col_indices is a dictionary, each value is processed to ensure
        compatibility with the data shape.
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

        if isinstance(row_indices, int) and nrow is not None and nrow > 0:
            row_indices = {"row": np.arange(row_indices, row_indices + nrow)}

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

        if isinstance(col_indices, int) and ncol is not None and ncol > 0:
            col_indices = {"col": np.arange(col_indices, col_indices + ncol)}

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
                    dim_other=nrow,
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
        nseries: int = 0,
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
        nrow : int, default 0
            The number of rows, by default 0. Must be defined if row_indices is
            not None.
        ncol : int, default 0
            The number of columns, by default 0. Must be defined if col_indices
            is not None.
        continuous : bool, default False
            Whether the rows and columns in row and col indices should be
            interpreted as continuous.
            If not continuous, the arrays given in row and col indices should
            correspond to coordinates by pixel.
            For DataCubes, the region must be continous. For DataFrames, it is
            assumed that the region is non-contiguous, by default False.
        nseries : int, default 0
           The number of series, by default 0.

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
            return (
                pd.MultiIndex.from_arrays([range(nseries)], names=["series"]),
                None,
                None,
            )

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
        # and the row and col indices were empty or not given
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
        """Sort the index of the data structure.

        This method wraps pandas' sort_index method and extends it to handle
        the uncertainty array and maintain the internal array structure.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to pandas' sort_index method.
        **kwargs : dict
            Keyword arguments to pass to pandas' sort_index method.
            Notable kwargs include:
            - inplace : bool, optional
                If True, perform operation in-place.
            - level : int or str, optional
                If index is a MultiIndex, sort on the given level.

        Returns
        -------
        self.__class__ or None
            If inplace=False, returns a new sorted object.
            If inplace=True, sorts in-place and returns None.

        Notes
        -----
        This method maintains the structure of the data and uncertainty arrays
        when sorting. It also ensures that convenience attributes are updated
        after sorting.

        See Also
        --------
        pandas.DataFrame.sort_index : The pandas method this wraps.
        """
        init_kwds = self.user_kwargs.copy()

        inplace = kwargs.pop("inplace", False)
        level = kwargs.get("level", None)
        pdobj = super(self._pd_class, self).sort_index(*args, **kwargs)
        sort_inds = np.argsort(self.index.get_level_values(level))

        if hasattr(pdobj, "columns"):
            series_inds = pdobj.columns.get_level_values("series")
        else:
            series_inds = None

        dfarray = pdobj.to_numpy()
        if ("axis" in kwargs and kwargs["axis"] in [0, "index"]) or (
            "axis" not in kwargs
        ):
            dfarray = dfarray.reshape((self.ntime, self.nrow, self.ncol))
        if inplace:
            self._array = dfarray

        if hasattr(self, "uncertainty") and bool(self.uncertainty):
            uncertainty_array = self.uncertainty.array
            uncertainty_array = uncertainty_array.reshape(self.shape)
            uncertainty_array = uncertainty_array[sort_inds]
            if series_inds is not None:
                uncertainty_array = uncertainty_array[:, series_inds]
            uncertainty_array = uncertainty_array.reshape(dfarray.shape)
            if inplace:
                self.uncertainty.array = uncertainty_array
            else:
                init_kwds["uncertainty"] = uncertainty_array
        if inplace:
            super(self._pd_class, self).__init__(pdobj)
            self._include_convenience_index()
            if hasattr(self, "columns"):
                self._include_convenience_columns()
        else:
            return self.__class__.from_pandas(
                pdobj, nrow=self.nrow, ncol=self.ncol, **init_kwds
            )


class MathMixin(IndexProcessorMixin):
    """Mixin class to add arithmetic to lightkurve data classes with uncertainty.

    See Also
    --------
    astropy.nddata.nduncertainty : astropy module from which uncertainty classes
        and operations have been derived.
    """

    _array = None
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
            # Collapsing functions should operate on pd.DataFrame-like array
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
            parentdata = self.to_numpy()
            uncertainty_copy.parent_nddata = parentdata
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
    def array(self):
        """Numpy array representation

        Cubes have shape (ntime, nrow, ncol)
        SeriesCollections have shape (ntime, nseries)
        and Series have shape (ntime)

        Returns
        -------
        np.ndarray
            An array representation of the data.

        Notes
        -----
        Uncertainties rely on parent data that are persistent and array-like.
        For Data classes, therefore, `array` must be stored in memory.
        This overwrites the on-call form defined in the ConvenienceMixin class.
        """
        return self._array

    @property
    def data(self):
        """Alias for self.array"""
        return self.array

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


class StatsMixin(MathMixin):

    """Generic mixin class for statistical methods in lightkurve data objects.


    This mixin provides common statistical methods such as mean, sum, std, var,
    min, max, and prod for lightkurve data objects. It also includes cumulative
    methods like cumsum, cummin, cummax, and cumprod.

    Attributes
    ----------
    ds_agg_func : str
        The aggregation function to use for downsampling, default is "sum".

    Methods
    -------
    median(**kwargs)
        Calculates the median of the data along a given axis.
    """

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
            if np_method in (np.argmin, np.argmax):
                if axis is None:
                    # returning unravelled index instead of flattened index
                    return np.unravel_index(np_method(self.array), self.array.shape)
                return np_method(self.to_numpy(), axis=axis)
            result, init_kwds = self._arithmetic(
                np_method, operand=None, data_axis=axis, uncertainty_axis=axis, **kwargs
            )
            init_kwds.update(self._get_math_kwargs())
            result = self._stats_post_process(result, axis=axis, **init_kwds)
            return result
            # pandas_method = getattr(super(self._pd_class, self), method_name)
            # return self._stats_post_process(pandas_method(*args, **kwargs), axis=axis)

        return _method

    def median(self, axis: Union[int, None] = None, **kwargs):
        """
        Get the median of the data along the given axis.

        This method calculates the median of the data and also estimates the
        uncertainty of the median based on the uncertainty of the mean.

        Parameters
        ----------
        axis : {0, 1, None}, default None
            Axis along which to calculate the median. If None, then calculate
            along each axis. If 0, then calculates the median for each pixel
            over all time steps and returns a DataFrame. If 1, then calculates
            the median of all pixels for each time step and returns a Series.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to numpy's median function.
            See np.ndarray.median for more details.

        Returns
        -------
        result : pd.DataFrame, lkData.Series, or float
            The median of the data along the specified axis.

        Raises
        ------
        ValueError
            If axis=2 is specified for Cubes, as it is not supported.

        Notes
        -----
        The uncertainty of the median is calculated based on the uncertainty of
        the mean, adjusted by a factor related to the number of data points.

        The efficiency of the variance of the median to the variance of the mean
        is calculated as (π * N) / (2 * (N - 1)), where N is the number of data
        points along the specified axis.

        See Also
        --------
        numpy.median : NumPy's median function used internally.
        """
        axis = kwargs.pop("axis", None)
        if axis == 2:
            raise (ValueError("For Cubes, axis=2 is not supported."))
        result = np.median(self.to_numpy(), axis=axis, **kwargs)

        # The uncertainty of the median is related to the uncertainty of the
        # mean by a scalar factor modulated by the number of data points
        _, init_kwds = self._arithmetic(
            np.mean, operand=None, data_axis=axis, uncertainty_axis=axis, **kwargs
        )
        uncertainty_mean = init_kwds["uncertainty"]

        N = self.shape[axis]
        # Efficiency of the variance of the median to the variance of the mean
        var_ratio = (np.pi * N) / (2 * (N - 1))
        init_kwds["uncertainty"].array = uncertainty_mean.array * var_ratio**0.5
        init_kwds.update(self._get_math_kwargs())
        result = self._stats_post_process(result, axis=axis, **init_kwds)
        return result

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
        Get the current display mode for values.

        Returns
        -------
        str
            The current display mode for values. Possible values are:
            - 'int': Display the raw integer values.
            - 'bitset': Display the values as sets of powers of 2.
            - 'detailed': Display the values as dictionaries mapping powers of 2 to their corresponding codes.

        Notes
        -----
        This property is used to control how values are displayed in the object's string representation
        and in any generated output (e.g., when using Jupyter notebooks).
        """
        return self._values_display

    @values_display.setter
    def values_display(self, value):
        allowed = {"int", "bitset", "detailed"}
        if value.lower() not in allowed:
            warn(f"Display must be one of {allowed}, defaulting to 'int'.")
            value = "int"
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
        Parse a bitset into a dictionary of corresponding codes.
        """
        # codes = self.breakdown(val)
        codes = list(val)
        codes.sort()
        str_codes = {code: self.codes.get(int(code), code) for code in codes}
        return str_codes

    def stylize_frame(self, df, **kwargs) -> Styler:
        """
        Overrides default to remove background gradient and to parse
        integers to a set of codes based on binary representation.
        """
        out = Styler(df)
        if "label" in kwargs:
            out = out.set_caption(kwargs.pop("label"))

        if (self._values_display == "bitset") or (self.codes == {}):
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
        # Get the values of the index on which to downsample
        index = self.index.get_level_values(level=level)
        index_names = list(self.index.names)

        sorted_inds = np.argsort(index)
        dfcopy = self.iloc[sorted_inds]

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
        if hasattr(dfcopy, "columns") and getattr(dfcopy, "columns") is not None:
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
            error = self.uncertainty.array[sorted_inds].reshape(self.shape)
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
                pass
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
        self,
        factor: Union[int, Tuple[int, int]] = 2,
        col_factor=None,
        row_name=None,
        col_name=None,
        **kwargs,
    ):
        """Spatially downsamples a DataCube by a given factor.

        Parameters
        ----------
        factor : int or tuple of int, default 2
            If a tuple is given, the first value will be used as the factor by
            which to reduce the size of the row axis and the second as the
            column factor.
            If factor is an integer and col_factor is also given, this is the
            factor by which to decrease the spatial resolution of the row axis.
            If col_factor is not given, this is the both the row and column factor.
        col_factor : int, optional
            Factor by which to decrease the spatial resolution of the column
            axis.
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
        if isinstance(factor, int):
            row_factor = kwargs.get("row_factor", factor)
            col_factor = col_factor or factor
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

        row_name = row_name or self.row_names[0]
        row = getattr(self, row_name)

        col_name = col_name or self.col_names[0]
        col = getattr(self, col_name)

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

        new_columns_left = self.columns.to_frame().groupby(
            [bin_edges_left_row, bin_edges_left_col], observed=False
        )
        if indexed:
            # If the old indices weren't positional, use min index for each bin for new index
            new_columns_left = new_columns_left.min().reset_index(drop=True)
        else:
            # If old indices were positional, use mean of bin for new index
            new_columns_left = new_columns_left.mean().reset_index(drop=True)

        new_columns = new_columns_left.set_index(self.columns.names).index[bin_mask]

        new_data = gb.agg(self.ds_agg_func)[bin_mask]

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
            new_data.T.to_numpy(),
            nrow=len(new_columns.get_level_values(row_name).unique()),
            ncol=len(new_columns.get_level_values(col_name).unique()),
            index=self.index,
            columns=new_columns,
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
        """Build a new instance of the class

        Parameters
        ----------
        new : Union[List, np.ndarray, Dict[str, Iterable]]
            New data with which to build  the new instance of the class.

        Returns
        -------
        type(self)
            New instance of the class with updated data and kwargs.
        """
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

    def fillna(self, *args, **kwargs):
        """Overwrite pandas method to return lk object"""
        pandas_method = getattr(super(self._pd_class, self), "fillna")
        new = pandas_method(*args, **kwargs)
        if hasattr(self, "ncol"):
            return self.from_pandas(
                new,
                nrow=self.nrow,
                ncol=self.ncol,
                **self.user_kwargs,
            )
        else:
            return self.from_pandas(
                new,
                **self.user_kwargs,
            )

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
    def ntime(self):
        """Number of cadences in the data."""
        return self.shape[0]

    @property
    def user_kwargs(self):
        """Keywords passed by the user"""
        return {key: getattr(self, key, None) for key in self._user_kwargs}
