"""Tests for lk bitset"""
import pytest
from lkdata.bitset import BitSet


def test_bitset_breakdown():
    """BitSet relies on breaking integers down into unique powers of 2"""
    assert BitSet.breakdown(0) == set()  # 0 -> empty set
    assert BitSet.breakdown(1) == {1}  # int -> set
    assert BitSet.breakdown(2) == {2}  # different int -> set
    assert BitSet.breakdown(3) == {1, 2}  # non-base int -> set of powers of 2
    assert BitSet.breakdown([1, 2, 3]) == {1, 2}  # iterable -> powers of 2
    assert BitSet.breakdown([1, 1, 2, 2, 3, 3, 3]) == {1, 2}  # repeats consolidated
    assert BitSet.breakdown("0b11") == {1, 2}  # binary int -> set of ints


def test_bitset_initialization():
    """Initialization tests"""
    # Test empty initialization
    bs = BitSet()
    assert len(bs) == 0
    assert str(bs) == "{}"

    # Test initialization with integer
    bs = BitSet(5)
    assert len(bs) == 2
    assert str(bs) == "{1, 4}"

    # Test initialization with list
    bs = BitSet([1, 2, 4])
    assert len(bs) == 3
    assert str(bs) == "{1, 2, 4}"


def test_bitset_operations():
    """basic set-like operations which return a new BitSet"""
    bs1 = BitSet([1, 2, 4])
    bs2 = BitSet([2, 4, 8])
    assert int(bs2) == 14
    assert set(bs2) == {2, 4, 8}

    # Test union
    assert str(bs1 | bs2) == "{1, 2, 4, 8}"
    assert str(bs1 | int(bs2)) == str(bs1 | bs2)
    assert str(bs1 | set(bs2)) == str(bs1 | bs2)
    assert str(bs1 + bs2) == str(bs1 | bs2)
    assert str(bs1.union(bs2)) == str(bs1 | bs2)

    # Test intersection
    assert str(bs1 & bs2) == "{2, 4}"
    assert str(bs1.intersection(bs2)) == str(bs1 & bs2)
    assert str(bs1 & int(bs2)) == str(bs1 & bs2)
    assert str(bs1 & set(bs2)) == str(bs1 & bs2)

    # Test difference
    assert str(bs1 - bs2) == "{1}"
    assert str(bs1.difference(bs2)) == str(bs1 - bs2)
    assert str(bs1 - int(bs2)) == str(bs1 - bs2)
    assert str(bs1 - set(bs2)) == str(bs1 - bs2)

    # Test symmetric difference
    assert str(bs1 ^ bs2) == "{1, 8}"
    assert str(bs1.symmetric_difference(bs2)) == str(bs1 ^ bs2)
    assert str(bs1 ^ int(bs2)) == str(bs1 ^ bs2)
    assert str(bs1 ^ set(bs2)) == str(bs1 ^ bs2)

    # Test is disjoint (no overlap)
    assert bs1.isdisjoint(96)
    assert bs1.isdisjoint({32, 64})
    assert bs1.isdisjoint(BitSet(96))


def test_bitset_methods():
    """Methods that act directly on a BitSet object"""
    bs = BitSet([1, 2, 4])

    # Test add
    bs.add(8)
    assert str(bs) == "{1, 2, 4, 8}"

    # Test remove
    bs.remove(2)
    assert str(bs) == "{1, 4, 8}"

    with pytest.raises(KeyError, match=r"KeyError: \{16\} not in"):
        bs.remove(17)

    # Test discard
    bs.discard({8, 16})  # Should not raise an error
    assert str(bs) == "{1, 4}"

    # Test clear
    bs.clear()
    assert len(bs) == 0


def test_bitset_comparisons():
    """Comparison operations -- test bitset against bitset/set/int"""
    bs1 = BitSet([1, 2, 4])
    bs2 = BitSet([2, 4])
    bs3 = BitSet([1, 2, 4])

    # not equal
    assert bs1 != bs2
    assert bs1 != int(bs2)
    assert bs1 != set(bs2)
    # equal
    assert bs1 == bs3
    assert bs1 == int(bs3)
    assert bs1 == set(bs3)
    # subset
    assert bs1 <= bs3
    assert bs2 <= bs1
    assert bs1 <= int(bs3)
    assert int(bs1) <= bs3
    assert bs1 <= set(bs3)
    assert set(bs1) <= bs3
    # superset
    assert bs1 >= bs3
    assert bs1 >= bs2
    assert bs1 >= int(bs3)
    assert int(bs1) >= bs3
    assert bs1 >= set(bs3)
    assert set(bs1) >= bs3
    # proper subset
    assert not (bs1 < bs3)
    assert not (bs1 < int(bs3))
    assert not (int(bs1) < bs3)
    assert not (bs1 < set(bs3))
    assert not (set(bs1) < bs3)
    assert bs2 < bs1
    assert bs2 < int(bs1)
    assert int(bs2) < bs1
    assert bs2 < set(bs1)
    assert set(bs2) < bs1
    # proper superset
    assert not (bs1 > bs3)
    assert not (bs1 > int(bs3))
    assert not (int(bs1) > bs3)
    assert not (bs1 > set(bs3))
    assert not (int(bs1) > bs3)
    assert bs1 > bs2
    assert bs1 > int(bs2)
    assert int(bs1) > bs2
    assert bs1 > set(bs2)
    assert set(bs1) > bs2


def test_bitset_boolean_operations():
    """Comparison against bool checks if bitset is empty"""
    # bitset bool value is True
    bs = BitSet([1, 2, 4])
    assert bool(bs)
    assert bs & True
    assert not (bs & False)
    assert bs | True
    assert bs | False

    # bbitset bool value is false
    bs = BitSet()
    assert not bs
    assert not (bs & True)
    assert not (bs & False)
    assert bs | True
    assert not (bs | False)


def test_bitset_conversion():
    bs = BitSet([1, 2, 4])

    assert int(bs) == 7
    assert bs.bin() == "0b111"
    assert set(bs) == {1, 2, 4}
