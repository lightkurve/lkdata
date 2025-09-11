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


def test_dataset_init(data_only, sample_dataset, ntime):
    """Test init, minimal input"""
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


def test_dataset_getitem_string(data_only):
    """Test keyword retreival"""
    assert isinstance(data_only["datacube"], DataCube)
    assert isinstance(data_only["dataseriescollection"], DataSeriesCollection)
    assert isinstance(data_only["dataseries"], DataSeries)


def test_dataset_getitem_slice(data_only):
    """Test time slice"""
    sliced = data_only[1:5]
    assert isinstance(sliced, DataSet)
    assert all(val.shape[0] == 4 for val in sliced.data_products.values())


def test_dataset_getitem_tuple(data_only):
    """Time and space slice"""
    subset = data_only[1:5, :]
    assert all(
        isinstance(val, (DataCube, DataSeriesCollection, DataSeries))
        for val in subset.data_products.values()
    )


def test_dataset_repr(data_only):
    """repr"""
    repr_str = repr(data_only)
    assert "Data Products:" in repr_str
    assert "Bool Products:" not in repr_str
    assert "Bitwise Products:" not in repr_str


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


def test_dataset_downsample(data_only):
    downsampled = data_only.downsample(nframes=5)
    assert all(val.shape[0] == 20 for val in downsampled.data_products.values())


def test_dataset_spatial_downsample(data_only):
    downsampled = data_only.spatial_downsample(factor=2)
    assert all(val.array.shape == (100, 5, 6) for val in downsampled.cubes.values())
    assert all(
        val.array.shape == (100, 120) for val in downsampled.series_collections.values()
    )


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
