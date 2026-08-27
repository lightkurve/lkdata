import warnings

import numpy as np
import pandas as pd
import pytest
from lkdata.dataset import (
    DataProducts,
    BoolProducts,
    BitwiseProducts,
    DataSet,
)
from lkdata.datacube import DataCube, BoolCube, BitwiseCube
from lkdata.seriescollection import (
    DataSeriesCollection,
    BoolSeriesCollection,
    BitwiseSeriesCollection,
)
from lkdata.dataseries import DataSeries, BoolSeries, BitwiseSeries


@pytest.fixture
def ntime():
    """ntime"""
    return 100


@pytest.fixture
def nrow():
    """nrow"""
    return 10


@pytest.fixture
def ncol():
    """ncol"""
    return 12


@pytest.fixture
def time_indices(ntime):
    """Dictionary of times"""
    time_inds = {
        "time": np.linspace(0, 10, ntime),
        "t1": np.linspace(2700, 2710, ntime),
        "t2": np.linspace(-5, 5, ntime),
        "t3": np.linspace(0, 1, ntime),
    }
    return time_inds


@pytest.fixture
def timeindex(time_indices):
    """Time indices as pandas MultiIndex"""
    index = pd.MultiIndex.from_arrays(list(time_indices.values()), time_indices.keys())
    return index


@pytest.fixture
def datacube(ntime, nrow, ncol):
    """DataCube"""
    return DataCube(
        np.random.rand(ntime, nrow, ncol), np.random.rand(ntime, nrow, ncol)
    )


@pytest.fixture
def dataseriescollection(ntime, nrow, ncol):
    """DataSeriesCollection"""
    return DataSeriesCollection(
        np.random.rand(ntime, nrow * ncol), np.random.rand(ntime, nrow * ncol)
    )


@pytest.fixture
def dataseries(ntime):
    """DataSeries"""
    return DataSeries(np.random.rand(ntime), np.random.rand(ntime))


@pytest.fixture
def boolcube(ntime, nrow, ncol):
    """BoolCube"""
    return BoolCube(np.random.choice((True, False), (ntime, nrow, ncol)))


@pytest.fixture
def boolseriescollection(ntime, nrow, ncol):
    """BoolSeriesCollection"""
    return BoolSeriesCollection(np.random.choice((True, False), (ntime, nrow * ncol)))


@pytest.fixture
def boolseries(ntime):
    """BoolSeries"""
    return BoolSeries(np.random.choice((True, False), (ntime)))


@pytest.fixture
def bitcube(ntime, nrow, ncol):
    """BitwiseCube"""
    return BitwiseCube(np.random.choice(64, (ntime, nrow, ncol)))


@pytest.fixture
def bitseriescollection(ntime, nrow, ncol):
    """BitwiseSeriesCollection"""
    return BitwiseSeriesCollection(np.random.choice(64, (ntime, nrow * ncol)))


@pytest.fixture
def bitseries(ntime):
    """BitwiseSeries"""
    return BitwiseSeries(np.random.choice(64, (ntime)))


@pytest.fixture
def data_only(datacube, dataseriescollection, dataseries):
    """DataSet with only data products, no error"""
    data = {
        "datacube": datacube,
        "dataseriescollection": dataseriescollection,
        "dataseries": dataseries,
    }
    return DataSet(data_products=data)


@pytest.fixture
def sample_dataset(
    datacube,
    dataseriescollection,
    dataseries,
    boolcube,
    boolseriescollection,
    boolseries,
    bitcube,
    bitseriescollection,
    bitseries,
):
    """DataSet with all product types"""
    data = {
        "datacube": datacube,
        "dataseriescollection": dataseriescollection,
        "dataseries": dataseries,
    }
    bools = {
        "boolcube": boolcube,
        "boolseriescollection": boolseriescollection,
        "boolseries": boolseries,
    }
    codes = {
        "bitcube": bitcube,
        "bitseriescollection": bitseriescollection,
        "bitseries": bitseries,
    }
    return DataSet(data_products=data, bool_products=bools, bitwise_products=codes)


# ─── DataSet init ────────────────────────────────────────────────────────────


def test_dataset_init(data_only, sample_dataset, ntime):
    """Test init with minimal and full input."""
    # Data only set, empty bool and bitwise products
    assert isinstance(data_only.data_products, DataProducts)
    assert len(data_only) == ntime
    for key in ["datacube", "dataseriescollection", "dataseries"]:
        assert key in data_only.data_products, f"{key} not in DataProducts"
    assert isinstance(data_only.bool_products, BoolProducts)
    assert not data_only.bool_products, "BoolProducts not empty"
    assert isinstance(data_only.bitwise_products, BitwiseProducts)
    assert not data_only.bitwise_products, "BitwiseProducts not empty"

    # DataSet with all the fixins'
    assert isinstance(sample_dataset.data_products, DataProducts)
    for key in ["datacube", "dataseriescollection", "dataseries"]:
        assert key in sample_dataset.data_products, f"{key} not in DataProducts"
    assert isinstance(sample_dataset.bool_products, BoolProducts)
    for key in ["boolcube", "boolseriescollection", "boolseries"]:
        assert key in sample_dataset.bool_products, f"{key} not in BoolProducts"
    assert isinstance(sample_dataset.bitwise_products, BitwiseProducts)
    for key in ["bitcube", "bitseriescollection", "bitseries"]:
        assert key in sample_dataset.bitwise_products, f"{key} not in BitwiseProducts"


# ─── DataSet.__getitem__ ─────────────────────────────────────────────────────


def test_dataset_getitem_string(data_only):
    """Test keyword retrieval."""
    assert isinstance(data_only["datacube"], DataCube)
    assert isinstance(data_only["dataseriescollection"], DataSeriesCollection)
    assert isinstance(data_only["dataseries"], DataSeries)


def test_dataset_getitem_slice(data_only):
    """Test time slice."""
    sliced = data_only[1:5]
    assert isinstance(sliced, DataSet)
    assert all(val.shape[0] == 4 for val in sliced.data_products.values())


def test_dataset_getitem_tuple(data_only):
    """Time and space slice."""
    subset = data_only[1:5, :]
    assert all(
        isinstance(val, (DataCube, DataSeriesCollection, DataSeries))
        for val in subset.data_products.values()
    )


# ─── DataSet.__repr__ ────────────────────────────────────────────────────────


def test_dataset_repr(data_only):
    """repr omits empty product sections."""
    repr_str = repr(data_only)
    assert "Data Products:" in repr_str
    assert "Bool Products:" not in repr_str
    assert "Bitwise Products:" not in repr_str


# ─── DataSet properties ──────────────────────────────────────────────────────


def test_dataset_cubes_property(data_only):
    cubes = data_only.cubes
    assert "datacube" in cubes
    assert isinstance(cubes["datacube"], DataCube)


def test_dataset_series_collections_property(data_only):
    series_collections = data_only.series_collections
    assert "dataseriescollection" in series_collections
    assert isinstance(series_collections["dataseriescollection"], DataSeriesCollection)


def test_dataset_series_property(data_only):
    series = data_only.series
    assert "dataseries" in series
    assert isinstance(series["dataseries"], DataSeries)


def test_dataset_contents_property(data_only):
    contents = data_only.contents
    assert "datacube" in contents
    assert "dataseriescollection" in contents
    assert "dataseries" in contents


# ─── DataSet methods ─────────────────────────────────────────────────────────


def test_dataset_downsample(data_only):
    downsampled = data_only.downsample(nframes=5)
    assert all(val.shape[0] == 20 for val in downsampled.data_products.values())


def test_dataset_spatial_downsample(data_only):
    downsampled = data_only.spatial_downsample(factor=2)
    assert all(val.array.shape == (100, 5, 6) for val in downsampled.cubes.values())
    assert all(
        val.array.shape == (100, 120) for val in downsampled.series_collections.values()
    )


# ─── DataSet.fold ────────────────────────────────────────────────────────────


def test_dataset_fold(data_only):
    folded = data_only.fold(period=2)
    assert "phase" in folded.index.names
    assert np.all(folded.index.get_level_values("phase") < 1)


def test_dataset_droplevel(data_only, ntime):
    data_only.index = pd.MultiIndex.from_arrays(
        [range(ntime), range(100, 100 + ntime)], names=["time", "category"]
    )
    dropped = data_only.droplevel(level="category")
    assert "category" not in dropped.index.names


# ─── DataSet setitem / setattr ───────────────────────────────────────────────


def test_dataset_user_kwargs(datacube):
    ds = DataSet(data_products={"data": datacube}, custom_param="test")
    assert ds.attrs == {"custom_param": "test"}


def test_setitem(sample_dataset, ntime, nrow, ncol):
    diff_data = DataCube(np.ones((ntime, nrow, ncol)), np.ones((ntime, nrow, ncol)))
    sample_dataset["datacube"] = diff_data
    assert (sample_dataset["datacube"] == 1).all(axis=None)

    with pytest.raises(TypeError):
        sample_dataset["datacube"] = "cat"


def test_setattr(sample_dataset):
    # Make sure it doesn't start with the new attribute
    assert not hasattr(sample_dataset, "cat")
    # Add it and make sure it's there
    sample_dataset.attrs["cat"] = "fluffy"
    assert sample_dataset.cat == "fluffy"
    assert "cat" in sample_dataset.attrs
    # Make sure it propagates to the contained products
    assert sample_dataset.data_products.cat == "fluffy"

    # Make sure it carries to derivative products
    assert sample_dataset[:10].cat == "fluffy"


ntime2, nrow2, ncol2 = 20, 4, 5
data = np.ones((ntime2, nrow2, ncol2))
err = np.ones((ntime2, nrow2, ncol2))


@pytest.fixture
def dc():
    return DataCube(data, uncertainty=err)


@pytest.fixture
def bc():
    return BoolCube(np.ones((ntime2, nrow2, ncol2), dtype=bool))


@pytest.fixture
def bic():
    return BitwiseCube(np.arange(ntime2 * nrow2 * ncol2).reshape(ntime2, nrow2, ncol2))


@pytest.fixture
def ds_fixture(dc, bc, bic):
    return DataSet(
        data_products={"cube": dc},
        bool_products={"mask": bc},
        bitwise_products={"flags": bic},
    )


# ─── DataSet.__getitem__ unrecognized string ──────────────────────────────────


def test_getitem_unrecognized_string_raises(ds_fixture):
    """Accessing a nonexistent key raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognized key"):
        _ = ds_fixture["does_not_exist"]


# ─── DataSet.__getitem__ with int ────────────────────────────────────────────


def test_getitem_int(ds_fixture):
    """Integer key returns a new DataSet with single time step."""
    result = ds_fixture[0]
    assert isinstance(result, DataSet)
    assert result.ntime == 1


# ─── DataSet.__getitem__ with list / array ────────────────────────────────────


def test_getitem_list(ds_fixture):
    """List of int indices returns a new DataSet with selected time steps."""
    result = ds_fixture[[0, 1, 2]]
    assert isinstance(result, DataSet)
    assert result.ntime == 3


def test_getitem_ndarray(ds_fixture):
    """NumPy array of indices returns a new DataSet."""
    result = ds_fixture[np.array([0, 5, 10])]
    assert isinstance(result, DataSet)
    assert result.ntime == 3


# ─── DataSet.__getitem__ with tuple (space + time) ────────────────────────────


def test_getitem_tuple_time_only(ds_fixture):
    """Tuple key with time only returns a DataSet."""
    result = ds_fixture[(slice(0, 5),)]
    assert isinstance(result, DataSet)


def test_getitem_tuple_time_and_space(ds_fixture):
    """Tuple key with time and spatial slice."""
    result = ds_fixture[0:5, :]
    assert isinstance(result, DataSet)


def test_getitem_tuple_too_many_raises(ds_fixture):
    """Tuple key with too many elements raises KeyError."""
    with pytest.raises(KeyError):
        _ = ds_fixture[0, 1, 2, 3]


# ─── ProductBundle._unpack_input with raw Iterables ──────────────────────────


def test_data_products_from_3d_array():
    """DataProducts initialized from a 3-D array creates a DataCube."""
    dp = DataProducts(data)
    assert len(dp) == 1
    key = list(dp.keys())[0]
    assert isinstance(dp[key], DataCube)


def test_data_products_from_2d_array():
    """DataProducts initialized from a 2-D array creates a DataSeriesCollection."""
    arr = np.ones((ntime2, nrow2 * ncol2))
    dp = DataProducts(arr)
    key = list(dp.keys())[0]
    assert isinstance(dp[key], DataSeriesCollection)


def test_data_products_from_1d_array():
    """DataProducts initialized from a 1-D array creates a DataSeries."""
    arr = np.ones(ntime2)
    dp = DataProducts(arr)
    key = list(dp.keys())[0]
    assert isinstance(dp[key], DataSeries)


# ─── ProductBundle._unpack_input with LkTypes ────────────────────────────────


def test_data_products_from_datacube(dc):
    """DataProducts initialized directly from a DataCube."""
    dp = DataProducts(dc)
    assert "DataCube" in dp


def test_bool_products_from_boolcube(bc):
    """BoolProducts initialized directly from a BoolCube."""
    bp = BoolProducts(bc)
    assert "BoolCube" in bp


def test_bitwise_products_from_bitwisecube(bic):
    """BitwiseProducts initialized directly from a BitwiseCube."""
    bip = BitwiseProducts(bic)
    assert "BitwiseCube" in bip


# ─── ProductBundle._unpack_input unsupported type ────────────────────────────


def test_data_products_unsupported_type_raises():
    """DataProducts with an unsupported type raises ValueError."""
    with pytest.raises(ValueError):
        DataProducts(42)


# ─── ProductBundle.__getitem__ (int/slice direct access) ─────────────────────


def test_productbundle_getitem_str(dc):
    """ProductBundle str key returns the stored value."""
    dp = DataProducts({"cube": dc})
    assert isinstance(dp["cube"], DataCube)


def test_productbundle_getitem_int(dc):
    """ProductBundle int key slices all contained products."""
    dp = DataProducts({"cube": dc})
    result = dp[0]
    assert isinstance(result, dict)
    assert "cube" in result


def test_productbundle_getitem_slice(dc):
    """ProductBundle slice key slices all contained products."""
    dp = DataProducts({"cube": dc})
    result = dp[0:5]
    assert isinstance(result, dict)


# ─── ProductBundle.apply ─────────────────────────────────────────────────────


def test_productbundle_apply(dc):
    """ProductBundle.apply applies a function to all contained products."""
    dp = DataProducts({"cube": dc})
    doubled = dp.apply(lambda x: x * 2)
    assert isinstance(doubled, DataProducts)
    assert (doubled["cube"] == 2).all(axis=None)


# ─── DataSet.ntime setter ────────────────────────────────────────────────────


def test_dataset_ntime_setter_non_empty_raises(ds_fixture):
    """Setting ntime on a DataSet with a non-empty index raises AttributeError."""
    with pytest.raises(AttributeError, match="Cannot set ntime"):
        ds_fixture.ntime = 99


# ─── DataSet.fold edge cases ─────────────────────────────────────────────────


def test_dataset_fold_invalid_period_raises(ds_fixture):
    """fold() with period <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="period"):
        ds_fixture.fold(period=0)


def test_dataset_fold_with_t0(ds_fixture):
    """fold() with explicit t0 subtracts it from time."""
    folded = ds_fixture.fold(period=5, t0=1)
    assert "phase" in folded.index.names


def test_dataset_fold_existing_label(ds_fixture):
    """fold() when 'phase' is already in the index drops and re-adds it."""
    folded = ds_fixture.fold(period=5)
    assert "phase" in folded.index.names
    # Fold again - 'phase' is now already in index names
    refolded = folded.fold(period=1, label="phase")
    assert "phase" in refolded.index.names


def test_dataset_fold_inplace(ds_fixture):
    """fold(inplace=True) modifies the DataSet in place and returns None."""
    result = ds_fixture.fold(period=5, inplace=True)
    assert result is None
    assert "phase" in ds_fixture.index.names


# ─── DataSet.describe_set ────────────────────────────────────────────────────


def test_dataset_describe_set(ds_fixture, capsys):
    """describe_set prints product information without raising."""
    ds_fixture.describe_set()
    captured = capsys.readouterr()
    assert "DataSet" in captured.out


def test_dataset_describe_set_with_attrs(dc, capsys):
    """describe_set prints user attrs when present."""
    ds = DataSet(data_products={"cube": dc}, my_param=42)
    ds.describe_set()
    assert "my_param" in capsys.readouterr().out


# ─── _check_attrs mismatch ────────────────────────────────────────────────────


def test_check_attrs_dimension_mismatch():
    """Adding a product with mismatched ntime raises ValueError."""
    dc1 = DataCube(np.ones((ntime2, nrow2, ncol2)))
    dc2 = DataCube(np.ones((ntime2 + 5, nrow2, ncol2)))
    with pytest.raises(ValueError):
        DataSet(data_products={"a": dc1, "b": dc2})


# ─── _build_data_product with different dims ─────────────────────────────────


def test_data_products_from_raw_with_kwargs():
    """DataProducts built from a raw array with cube-building kwargs."""
    dp = DataProducts(
        data,
        nrow=nrow2,
        ncol=ncol2,
    )
    key = list(dp.keys())[0]
    assert isinstance(dp[key], DataCube)


# ─── DataSet __repr__ ────────────────────────────────────────────────────────


def test_dataset_repr_with_all_products(ds_fixture):
    """DataSet repr shows all product types."""
    r = repr(ds_fixture)
    assert "Data Products" in r
    assert "Bool Products" in r
    assert "Bitwise Products" in r


def test_dataset_repr_with_user_kwargs(dc):
    """DataSet repr shows user kwargs."""
    ds = DataSet(data_products={"cube": dc}, sensor="TESS")
    r = repr(ds)
    assert "Properties" in r


# ─── ProductBundle IndexError warning ────────────────────────────────────────


def test_productbundle_getitem_index_error_warns(dc):
    """ProductBundle warns when a product can't be sliced (out-of-bounds IndexError)."""
    dc2 = DataCube(np.ones((ntime2, nrow2, ncol2)))
    dp = DataProducts({"a": dc, "b": dc2})
    # Requesting index ntime (out of bounds for both) triggers IndexError → warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = dp[ntime2]
    assert len(w) > 0
    assert "a" in result
