from collections.abc import MutableSet, Hashable
from typing import Iterable, Union
import math


class BitSet(MutableSet, Hashable):
    """A data type for bitwise objects.

    This datatype combines the utility of sets and the succinct representation
    of integer numbers as used for data quality flags in astronomical data.
    Integers are essentially treated as sets of their constituent powers of 2.
    """

    name = "bitwise"
    type = int

    __hash__ = MutableSet._hash

    wrap_wo_mod = (
        "clear",
        "copy",
        "pop",
    )

    wrap_w_breakdown_bools = (
        "isdisjoint",
        "issubset",
        "issuperset",
    )

    wrap_w_breakdown = (
        "difference",
        "discard",
        "intersection",
        "symmetric_difference",
        "union",
    )

    wrap_w_breakdown_mod = (
        "difference_update",
        "intersection_update",
        "remove",
        "symmetric_difference_update",
        "update",
    )

    def __new__(cls, iterable=None):
        selfobj = super(BitSet, cls).__new__(BitSet)

        selfobj._set = set() if iterable is None else BitSet.breakdown(iterable)

        for method_name in cls.wrap_wo_mod:
            setattr(selfobj, method_name, cls._wrap_method(method_name, selfobj))
        for method_name in cls.wrap_w_breakdown_mod:
            setattr(
                selfobj,
                method_name,
                cls._wrap_breakdown_method(method_name, selfobj, return_type="none"),
            )
        for method_name in cls.wrap_w_breakdown:
            setattr(
                selfobj,
                method_name,
                cls._wrap_breakdown_method(method_name, selfobj, return_type="bitset"),
            )
        for method_name in cls.wrap_w_breakdown_bools:
            setattr(
                selfobj,
                method_name,
                cls._wrap_breakdown_method(method_name, selfobj, return_type="bool"),
            )
        return selfobj

    def __add__(self, value):
        valset = self.breakdown(value)
        valset.update(self._set)
        return BitSet(valset)

    def __and__(self, other):
        if isinstance(other, bool):
            return bool(self) and other
        else:
            return BitSet(self._bitset_method("__and__", other))

    def __bool__(self):
        # True if sum of values is nonzero
        return bool(int(self._set))

    def __repr__(self):
        return f"BitSet {repr(self._set)}"

    def __getattr__(self, attr):
        return getattr(self._set, attr)

    def __contains__(self, val):
        valset = self.breakdown(val)
        return valset.issubset(self._set)

    def __int__(self):
        return sum(self._set)

    def __iter__(self):
        return iter(self._set)

    def __eq__(self, value):
        if isinstance(value, bool):
            return bool(self) == value
        return self._bitset_method("__eq__", value)

    def __ge__(self, value):
        return self._bitset_method("__ge__", value)

    def __gt__(self, value):
        return self._bitset_method("__gt__", value)

    def __le__(self, value):
        return self._bitset_method("__le__", value)

    def __len__(self):
        return len(self._set)

    def __lt__(self, value):
        return self._bitset_method("__lt__", value)

    def __ne__(self, value):
        if isinstance(value, bool):
            return bool(self) != value
        return self._bitset_method("__ne__", value)

    def __or__(self, other):
        if isinstance(other, bool):
            return bool(self) or other
        else:
            return BitSet(self._bitset_method("__or__", other))

    def __rand__(self, other):
        if isinstance(other, bool):
            return bool(self) and other
        else:
            return BitSet(self._bitset_method("__rand__", other))

    def __ror__(self, other):
        if isinstance(other, bool):
            return bool(self) or other
        else:
            return BitSet(self._bitset_method("__ror__", other))

    def __rxor__(self, other):
        if isinstance(other, bool):
            return other.__rxor__(bool(self))
        else:
            return BitSet(self._bitset_method("__rxor__", other))

    def __str__(self):
        return str(self._set)

    def __sub__(self, value):
        valset = self.breakdown(value)
        new = self._set - valset
        return BitSet(new)

    def __xor__(self, other):
        if isinstance(other, bool):
            return bool(self).__xor__(other)
        else:
            return BitSet(self._bitset_method("__xor__", other))

    def _bitset_method(self, method_name, value):
        valset = BitSet.breakdown(value)
        result = getattr(self._set, method_name)(valset)
        return result

    @classmethod
    def _wrap_method(cls, method_name, obj):
        def method(*args, **kwargs):
            result = getattr(obj._set, method_name)(*args, **kwargs)
            return BitSet(result)

        return method

    @classmethod
    def _wrap_breakdown_method(cls, method_name, obj, return_type="none"):
        def method(value):
            valset = BitSet.breakdown(value)
            result = getattr(obj._set, method_name)(valset)
            if return_type == "none":
                return None
            elif return_type == "bitset":
                return BitSet(result)
            elif return_type == "bool":
                return result

        return method

    def add(self, value):
        valset = self.breakdown(value)
        self._set.update(valset)

    def bin(self):
        return bin(int(self))

    @staticmethod
    def breakdown(item: Union[int, Iterable]):
        """Breaks down a given item into a single set of bitwise components

        Parameters
        ----------
        item : Union[int, str, Iterable]
            An integer, binary integer, or collection of integers to be broken down
        """

        codes = set()
        if isinstance(item, Iterable):
            # Recursion loop until getting to individual values
            for val in item:
                codes.update(BitSet.breakdown(val))
            return codes

        # Ensure properly formatted binary representation
        if isinstance(item, str):
            item = int(item, 2)
        asbin = bin(item)
        for pos, b in enumerate(asbin[:1:-1]):
            if int(b):
                codes.add(2 ** (pos))

        return codes

    def discard(self, value):
        valset = self.breakdown(value)
        self._set = self._set - valset


class LkFloat:
    """
    A class representing a float value with associated error.

    This class allows for mathematical operations while propagating errors.
    """

    def __init__(self, value, error=0.0):
        """
        Initialize LkFloat with a value and an optional error.

        Args:
            value (float): The central value.
            error (float, optional): The associated error. Defaults to 0.0.
        """
        self.value = float(value)
        self.error = abs(float(error))

    def __repr__(self):
        """Return a string representation of the LkFloat object."""
        return f"LkFloat({self.value}, {self.error})"

    def __str__(self):
        """Return a human-readable string representation of the LkFloat object."""
        return f"{self.value} ± {self.error}"

    def __add__(self, other):
        """
        Add two LkFloat objects or an LkFloat and a number.

        Error is propagated using quadrature addition.
        """
        if isinstance(other, LkFloat):
            new_value = self.value + other.value
            new_error = math.sqrt(self.error**2 + other.error**2)
            return LkFloat(new_value, new_error)
        return LkFloat(self.value + other, self.error)

    def __radd__(self, other):
        """Handle addition when LkFloat is the right operand."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two LkFloat objects or subtract a number from an LkFloat.

        Error is propagated using quadrature addition.
        """
        if isinstance(other, LkFloat):
            new_value = self.value - other.value
            new_error = math.sqrt(self.error**2 + other.error**2)
            return LkFloat(new_value, new_error)
        return LkFloat(self.value - other, self.error)

    def __rsub__(self, other):
        """Handle subtraction when LkFloat is the right operand."""
        return LkFloat(other, 0) - self

    def __mul__(self, other):
        """
        Multiply two LkFloat objects or multiply an LkFloat by a number.

        Error is propagated using the product rule.
        """
        if isinstance(other, LkFloat):
            new_value = self.value * other.value
            new_error = math.sqrt(
                (self.error * other.value) ** 2 + (other.error * self.value) ** 2
            )
            return LkFloat(new_value, new_error)
        return LkFloat(self.value * other, abs(self.error * other))

    def __rmul__(self, other):
        """Handle multiplication when LkFloat is the right operand."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divide two LkFloat objects or divide an LkFloat by a number.

        Error is propagated using the quotient rule.
        """
        if isinstance(other, LkFloat):
            new_value = self.value / other.value
            new_error = math.sqrt(
                (self.error / other.value) ** 2
                + (other.error * self.value / other.value**2) ** 2
            )
            return LkFloat(new_value, new_error)
        return LkFloat(self.value / other, abs(self.error / other))

    def __rtruediv__(self, other):
        """Handle division when LkFloat is the right operand."""
        return LkFloat(other, 0) / self

    def __pow__(self, other):
        """
        Raise LkFloat to a power (either another LkFloat or a number).

        Error is propagated using the general formula for error propagation in exponentiation.
        """
        if isinstance(other, LkFloat):
            new_value = self.value**other.value
            new_error = abs(new_value) * math.sqrt(
                (self.error / self.value) ** 2
                + (other.error * math.log(abs(self.value))) ** 2
            )
            return LkFloat(new_value, new_error)
        new_value = self.value**other
        new_error = abs(new_value * other * self.error / self.value)
        return LkFloat(new_value, new_error)

    def __rpow__(self, other):
        """Handle exponentiation when LkFloat is the right operand."""
        return LkFloat(other, 0) ** self

    def __eq__(self, other):
        """
        Check if two LkFloat objects are equal within their error ranges.

        Two LkFloats are considered equal if their ranges overlap.
        """
        if isinstance(other, LkFloat):
            return abs(self.value - other.value) <= (self.error + other.error)
        return abs(self.value - other) <= self.error

    def __ne__(self, other):
        """Check if two LkFloat objects are not equal."""
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Check if one LkFloat is less than another LkFloat or number.

        Considers both value and error range.
        """
        if isinstance(other, LkFloat):
            return self.value < other.value and not self.__eq__(other)
        return self.value < other and not self.__eq__(other)

    def __le__(self, other):
        """Check if one LkFloat is less than or equal to another LkFloat or number."""
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        """Check if one LkFloat is greater than another LkFloat or number."""
        return not self.__le__(other)

    def __ge__(self, other):
        """Check if one LkFloat is greater than or equal to another LkFloat or number."""
        return not self.__lt__(other)
