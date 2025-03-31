from collections.abc import MutableSet, Hashable
from typing import Iterable, Union


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
        return bool(self._set)

    def __repr__(self):
        lset = list(self._set)
        lset.sort()
        return f"BitSet {repr(lset).replace('[', '{').replace(']', '}')}"

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
        if self._set:
            ordered = list(self._set)
            ordered.sort()
            return str(ordered).replace("[", "{").replace("]", "}")
        return "{}"

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
