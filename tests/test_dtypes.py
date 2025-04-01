from lkdata.dtypes import BitSet
import pytest


def test_bitset_initialization():
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
    bs1 = BitSet([1, 2, 4])
    bs2 = BitSet([2, 4, 8])

    # Test union
    assert str(bs1 | bs2) == "{1, 2, 4, 8}"

    # Test intersection
    assert str(bs1 & bs2) == "{2, 4}"

    # Test difference
    assert str(bs1 - bs2) == "{1}"

    # Test symmetric difference
    assert str(bs1 ^ bs2) == "{1, 8}"


def test_bitset_methods():
    bs = BitSet([1, 2, 4])

    # Test add
    bs.add(8)
    assert str(bs) == "{1, 2, 4, 8}"

    # Test remove
    bs.remove(2)
    assert str(bs) == "{1, 4, 8}"

    with pytest.raises(KeyError, match="KeyError: \{16\} not in"):
        bs.remove(17)

    # Test discard
    bs.discard(16)  # Should not raise an error
    assert str(bs) == "{1, 4, 8}"

    # Test clear
    bs.clear()
    assert len(bs) == 0


def test_bitset_comparisons():
    bs1 = BitSet([1, 2, 4])
    bs2 = BitSet([2, 4, 8])
    bs3 = BitSet([1, 2, 4])

    assert bs1 != bs2
    assert bs1 == bs3
    assert bs1 <= bs3
    assert bs1 >= bs3
    assert not (bs1 < bs3)
    assert not (bs1 > bs3)


def test_bitset_boolean_operations():
    bs = BitSet([1, 2, 4])

    assert bool(bs)
    assert not bool(BitSet())

    assert bs & True
    assert not (bs & False)
    assert bs | True
    assert bs | False


def test_bitset_conversion():
    bs = BitSet([1, 2, 4])

    assert int(bs) == 7
    assert bs.bin() == "0b111"
