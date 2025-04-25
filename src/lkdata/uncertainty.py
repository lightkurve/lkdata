# This code is modified from astropy.nddata.nduncertainty
# and is distributed under the following MIT license:
## Licensed under a 3-clause BSD style license - see LICENSE.rst
## Copyright (c) 2011-2024, Astropy Developers
## All rights reserved.

import weakref
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

__all__ = [
    "MissingDataAssociationException",
    "IncompatibleUncertaintiesException",
    "NDUncertainty",
    "Uncertainty",
]

# mapping from collapsing operations to the complementary methods used for `to_variance`
collapse_to_variance_mapping = {
    np.sum: np.square,
    np.mean: np.square,
}


def _move_preserved_axes_first(arr, preserve_axes):
    # When collapsing an ND array and preserving M axes, move the
    # preserved axes to the first M axes of the output. For example,
    # if arr.shape == (6, 5, 4, 3, 2) and we're preserving axes (1, 2),
    # then the output should have shape (20, 6, 3, 2). Axes 1 and 2 have
    # shape 5 and 4, so we take their product and put them both in the zeroth
    # axis.
    zeroth_axis_after_reshape = np.prod(np.array(arr.shape)[list(preserve_axes)])
    collapse_axes = [i for i in range(arr.ndim) if i not in preserve_axes]
    return arr.reshape(
        [zeroth_axis_after_reshape] + np.array(arr.shape)[collapse_axes].tolist()
    )


def _unravel_preserved_axes(arr, collapsed_arr, preserve_axes):
    # After reshaping an array with _move_preserved_axes_first and collapsing
    # the result, convert the reshaped first axis back into the shape of each
    # of the original preserved axes.
    # For example, if arr.shape == (6, 5, 4, 3, 2) and we're preserving axes (1, 2),
    # then the output of _move_preserved_axes_first should have shape (20, 6, 3, 2).
    # This method unravels the first axis in the output *after* a collapse, so the
    # output with shape (20,) becomes shape (5, 4).
    if collapsed_arr.ndim != len(preserve_axes):
        arr_shape = np.array(arr.shape)
        return collapsed_arr.reshape(arr_shape[np.asarray(preserve_axes)])
    return collapsed_arr


def from_variance_for_mean(x, axis):
    if axis is None:
        # do operation on all dimensions:
        denom = np.ma.count(x)
    else:
        denom = np.ma.count(x, axis)
    return np.sqrt(np.ma.sum(x, axis)) / denom


# mapping from collapsing operations to the complementary methods used for `from_variance`
collapse_from_variance_mapping = {
    np.sum: lambda x, axis: np.sqrt(np.ma.sum(x, axis)),
    np.mean: from_variance_for_mean,
}


class IncompatibleUncertaintiesException(Exception):  # pragma: no cover
    """This exception should be used to indicate cases in which uncertainties
    with two different classes can not be propagated.
    """


class MissingDataAssociationException(Exception):  # pragma: no cover
    """This exception should be used to indicate that an uncertainty instance
    has not been associated with a parent `~astropy.nddata.NDData` object.
    """


class NDUncertainty(metaclass=ABCMeta):  # pragma: no cover
    """This is the metaclass for uncertainty classes used with `NDData`.

    Parameters
    ----------
    array : any type, optional
        The array or value (the parameter name is due to historical reasons) of
        the uncertainty. `numpy.ndarray` or `NDUncertainty` subclasses are recommended.
        If the `array` is `list`-like or `numpy.ndarray`-like it will be cast
        to a plain `numpy.ndarray`.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the `array` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class as ``array`` if their
        ``uncertainty_type`` is different.
    """

    def __init__(self, array=None, copy=True):
        if isinstance(array, NDUncertainty):
            # Given an NDUncertainty class or subclass check that the type
            # is the same.
            if array.uncertainty_type != self.uncertainty_type:
                raise IncompatibleUncertaintiesException
            array = array.array

        if copy:
            array = deepcopy(array)

        self.array = array
        self.parent_nddata = None  # no associated NDData - until it is set!

    def __bool__(self):
        return (self.array is not None) and (len(np.atleast_1d(self.array)) > 0)

    @property
    @abstractmethod
    def uncertainty_type(self):
        """`str` : Short description of the type of uncertainty.

        Defined as abstract property so subclasses *have* to override this.
        """
        return None

    @property
    def supports_correlated(self):
        """`bool` : Supports uncertainty propagation with correlated uncertainties?

        .. versionadded:: 1.2
        """
        return False

    @property
    def array(self):
        """`numpy.ndarray` : the uncertainty's value."""
        return self._array

    @array.setter
    def array(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = np.asarray(value)
        self._array = value

    @property
    def parent_nddata(self):
        """`NDData` : reference to `NDData` instance with this uncertainty.

        In case the reference is not set uncertainty propagation will not be
        possible since propagation might need the uncertain data besides the
        uncertainty.
        """
        no_parent_message = "uncertainty is not associated with an NDData object"

        try:
            parent = self._parent_nddata
        except AttributeError:
            raise MissingDataAssociationException(no_parent_message)
        else:
            if parent is None:
                raise MissingDataAssociationException(no_parent_message)
            else:
                # The NDData is saved as weak reference so we must call it
                # to get the object the reference points to. However because
                # we have a weak reference here it's possible that the parent
                # was deleted because its reference count dropped to zero.
                if isinstance(self._parent_nddata, weakref.ref):
                    resolved_parent = self._parent_nddata()
                    # if resolved_parent is None:
                    # log.info(parent_lost_message)
                    return resolved_parent
                else:
                    # log.info("parent_nddata should be a weakref to an NDData object.")
                    return self._parent_nddata

    @parent_nddata.setter
    def parent_nddata(self, value):
        if value is not None and not isinstance(value, weakref.ref):
            # Save a weak reference on the uncertainty that points to this
            # instance of NDData. Direct references should NOT be used:
            # https://github.com/astropy/astropy/pull/4799#discussion_r61236832
            value = weakref.ref(value)
        # Set _parent_nddata here and access below with the property because value
        # is a weakref
        self._parent_nddata = value

    def __repr__(self):
        prefix = self.__class__.__name__ + "("
        try:
            body = np.array2string(self.array, separator=", ", prefix=prefix)
        except AttributeError:
            # In case it wasn't possible to use array2string
            body = str(self.array)
        return f"{prefix}{body})"

    def __getstate__(self):
        # Because of the weak reference the class wouldn't be picklable.
        try:
            return self._array, self.parent_nddata
        except MissingDataAssociationException:
            # In case there's no parent
            return self._array, None

    def __setstate__(self, state):
        if len(state) != 2:
            raise TypeError("The state should contain 2 items.")
        self._array = state[0]

        parent = state[1]
        if parent is not None:
            parent = weakref.ref(parent)
        self._parent_nddata = parent

    def __getitem__(self, item):
        """Normal slicing on the array, return a reference."""
        return self.__class__(self.array[item], copy=False)

    def propagate(self, operation, other_nddata, result_data, correlation, axis=None):
        """Calculate the resulting uncertainty given an operation on the data.

        .. versionadded:: 1.2

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `NDData`. Supported are
            `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide` (or `numpy.divide`).

        other_nddata : `NDData` instance
            The second operand in the arithmetic operation.

        result_data : ndarray
            The result of the arithmetic operations on the data.

        correlation : `numpy.ndarray` or number
            The correlation (rho) is defined between the uncertainties in
            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means
            uncorrelated operands.

        axis : int or tuple of ints, optional
            Axis over which to perform a collapsing operation.

        Returns
        -------
        resulting_uncertainty : `NDUncertainty` instance
            Another instance of the same `NDUncertainty` subclass containing
            the uncertainty of the result.

        Raises
        ------
        ValueError
            If the ``operation`` is not supported or if correlation is not zero
            but the subclass does not support correlated uncertainties.

        Notes
        -----
        First this method checks if a correlation is given and the subclass
        implements propagation with correlated uncertainties.
        Then the second uncertainty is converted (or an Exception is raised)
        to the same class in order to do the propagation.
        Then the appropriate propagation method is invoked and the result is
        returned.
        """
        # Check if the subclass supports correlation
        if not self.supports_correlated:
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError(
                    f"{type(self).__name__} does not support uncertainty propagation"
                    " with correlation."
                )

        if other_nddata is not None:
            # Get the other uncertainty
            other_uncert = other_nddata.uncertainty

            if operation.__name__ == "add":
                result = self._propagate_add(other_uncert, result_data, correlation)
            elif operation.__name__ == "subtract":
                result = self._propagate_subtract(
                    other_uncert, result_data, correlation
                )
            elif operation.__name__ == "multiply":
                result = self._propagate_multiply(
                    other_uncert, result_data, correlation
                )
            elif operation.__name__ in ["true_divide", "divide"]:
                result = self._propagate_divide(other_uncert, result_data, correlation)
            else:
                raise ValueError(f"unsupported operation: {operation.__name__}")
        else:
            # assume this is a collapsing operation:
            result = self._propagate_collapse(operation, axis)

        return self.__class__(result, copy=False)

    @abstractmethod
    def _propagate_add(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_divide(self, other_uncert, result_data, correlation):
        return None

    @abstractmethod
    def _propagate_collapse(self, operation, axis):
        return None

    def reshape(self, *args, **kwargs):
        """See NDUncertainty.array.reshape()"""
        return self.__class__(self.array.reshape(*args, **kwargs))

    @property
    def shape(self):
        """NDUncertainty.array.shape"""
        return self.array.shape


class _VariancePropagationMixin:
    """
    Propagation of uncertainties for variances, also used to perform error
    propagation for variance-like uncertainties (standard deviation and inverse
    variance).
    """

    def _propagate_collapse(self, numpy_operation, axis=None):
        """
        Error propagation for collapse operations on variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        numpy_op : function
            Numpy operation like `np.sum` or `np.max` to use in the collapse

        subtract : bool, optional
            If ``True``, propagate for subtraction, otherwise propagate for
            addition.

        axis : tuple, optional
            Axis on which to compute collapsing operations.
        """

        if self.array is not None:
            # Formula: sigma**2 = dA

            if numpy_operation in [np.min, np.max]:
                # Find the indices of the min/max in parent data along each axis,
                # return the uncertainty at the corresponding entry:
                return self._get_err_at_extremum(numpy_operation, axis=axis)

            # np.sum and np.mean operations use similar pattern
            # to `_propagate_add_sub`, for example:
            else:
                # lookup the mapping for to_variance and from_variance for this
                # numpy operation:
                to_variance = collapse_to_variance_mapping.get(
                    numpy_operation, lambda x: x
                )
                from_variance = collapse_from_variance_mapping.get(
                    numpy_operation, numpy_operation
                )

                this = to_variance(self.array)

                return from_variance(this, axis=axis)

    def _get_err_at_extremum(self, extremum, axis):
        """
        Return the value of the ``uncertainty`` array at the indices
        which satisfy the ``extremum`` function applied to the ``measurement`` array,
        where we expect ``extremum`` to be np.argmax or np.argmin, and
        we expect a two-dimensional output.

        Assumes the ``measurement`` and ``uncertainty`` array dimensions
        are ordered such that the zeroth dimension is the one to preserve.
        For example, if you start with array with shape (a, b, c), this
        function applies the ``extremum`` function to the last two dimensions,
        with shapes b and c.

        This operation is difficult to cast in a vectorized way. Here
        we implement it with a list comprehension, which is likely not the
        most performant solution.
        """
        if axis is not None and not hasattr(axis, "__len__"):
            # this is a single axis:
            axis = [axis]

        if extremum is np.min:
            arg_extremum = np.ma.argmin
        elif extremum is np.max:
            arg_extremum = np.ma.argmax

        all_axes = np.arange(self.array.ndim)

        if axis is None:
            # collapse over all dimensions
            ind = arg_extremum(np.asanyarray(self.parent_nddata).ravel())
            return self.array.ravel()[ind]

        # collapse an ND array over arbitrary dimensions:
        preserve_axes = [ax for ax in all_axes if ax not in axis]
        meas = _move_preserved_axes_first(self.parent_nddata, preserve_axes)
        err = _move_preserved_axes_first(self.array, preserve_axes)

        result = np.array(
            [e[np.unravel_index(arg_extremum(m), m.shape)] for m, e in zip(meas, err)]
        )

        return _unravel_preserved_axes(
            self.parent_nddata.data,
            result,
            preserve_axes,
        )

    def _propagate_add_sub(
        self,
        other_uncert,
        _,  # arg unused for uncertainty propagation in addition/subtraction
        correlation,
        subtract=False,
        to_variance=lambda x: x,
        from_variance=lambda x: x,
    ):
        """
        Error propagation for addition or subtraction of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        subtract : bool, optional
            If ``True``, propagate for subtraction, otherwise propagate for
            addition.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """
        if subtract:
            correlation_sign = -1
        else:
            correlation_sign = 1

        if other_uncert.array is not None:
            other = to_variance(other_uncert.array)
        else:
            other = 0

        if self.array is not None:
            # Formula: sigma**2 = dA
            this = to_variance(self.array)
        else:
            this = 0

        # Formula: sigma**2 = dA + dB +/- 2*cor*sqrt(dA*dB)
        # Formula: sigma**2 = sigma_other + sigma_self +/- 2*cor*sqrt(dA*dB)
        #     (sign depends on whether addition or subtraction)

        # Determine the result depending on the correlation
        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = 2 * correlation * np.sqrt(this * other)
            result = this + other + correlation_sign * corr
        else:
            result = this + other

        return from_variance(result)

    def _propagate_multiply_divide(
        self,
        other_uncert,
        _,  # arg unused
        correlation,
        divide=False,
        to_variance=lambda x: x,
        from_variance=lambda x: x,
    ):
        """
        Error propagation for multiplication or division of variance or
        variance-like uncertainties. Uncertainties are calculated using the
        formulae for variance but can be used for uncertainty convertible to
        a variance.

        Parameters
        ----------
        other_uncert : `~astropy.nddata.NDUncertainty` instance
            The uncertainty, if any, of the other operand.

        result_data : `~astropy.nddata.NDData` instance
            The results of the operation on the data.

        correlation : float or array-like
            Correlation of the uncertainties.

        divide : bool, optional
            If ``True``, propagate for division, otherwise propagate for
            multiplication.

        to_variance : function, optional
            Function that will transform the input uncertainties to variance.
            The default assumes the uncertainty is the variance.

        from_variance : function, optional
            Function that will convert from variance to the input uncertainty.
            The default assumes the uncertainty is the variance.
        """
        if divide:
            correlation_sign = -1
        else:
            correlation_sign = 1

        if other_uncert.array is not None:
            d_b = to_variance(other_uncert.array)
            # Formula: sigma**2 = |A|**2 * d_b
            right = np.abs(self.parent_nddata**2 * d_b)
        else:
            right = 0

        if self.array is not None:
            # Just the reversed case
            d_a = to_variance(self.array)
            # Formula: sigma**2 = |B|**2 * d_a
            left = np.abs(other_uncert.parent_nddata**2 * d_a)
        else:
            left = 0

        # Multiplication
        #
        # The fundamental formula is:
        #   sigma**2 = |AB|**2*(d_a/A**2+d_b/B**2+2*sqrt(d_a)/A*sqrt(d_b)/B*cor)
        #
        # This formula is not very handy since it generates NaNs for every
        # zero in A and B. So we rewrite it:
        #
        # Multiplication Formula:
        #   sigma**2 = (d_a*B**2 + d_b*A**2 + (2 * cor * ABsqrt(dAdB)))
        #   sigma**2 = (left + right + (2 * cor * ABsqrt(dAdB)))
        #
        # Division
        #
        # The fundamental formula for division is:
        #   sigma**2 = |A/B|**2*(d_a/A**2+d_b/B**2-2*sqrt(d_a)/A*sqrt(d_b)/B*cor)
        #
        # As with multiplication, it is convenient to rewrite this to avoid
        # nans where A is zero.
        #
        # Division formula (rewritten):
        #   sigma**2 = d_a/B**2 + (A/B)**2 * d_b/B**2
        #                   - 2 * cor * A *sqrt(dAdB) / B**3
        #   sigma**2 = d_a/B**2 + (A/B)**2 * d_b/B**2
        #                   - 2*cor * sqrt(d_a)/B**2  * sqrt(d_b) * A / B
        #   sigma**2 = multiplication formula/B**4 (and sign change in
        #               the correlation)

        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = (
                2
                * correlation
                * np.sqrt(d_a * d_b)
                * self.parent_nddata
                * other_uncert.parent_nddata
            )
        else:
            corr = 0

        if divide:
            return from_variance(
                (left + right + correlation_sign * corr) / other_uncert.parent_nddata**4
            )
        else:
            return from_variance(left + right + correlation_sign * corr)


class Uncertainty(_VariancePropagationMixin, NDUncertainty):
    """Standard deviation uncertainty assuming first order gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `Uncertainty`.
    Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    `Uncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, Uncertainty
        >>> ndd = NDData([1,2,3],
        ...              uncertainty=Uncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        Uncertainty([0.1, 0.1, 0.1])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = Uncertainty([0.2], copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        Uncertainty([0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        Uncertainty(2)

    Note
    ----
    This class was originally StdDevUncertainty from
    astropy.nddata.nduncertainty and has been renamed Uncertainty for lkdata
    """

    @property
    def supports_correlated(self):
        """`True` : `Uncertainty` allows to propagate correlated \
                    uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """``"std"`` : `Uncertainty` implements standard deviation."""
        return "std"

    def _propagate_add(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=False,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=True,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=False,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=True,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_collapse(self, numpy_operation, axis):
        # defer to _VariancePropagationMixin
        return super()._propagate_collapse(numpy_operation, axis=axis)
