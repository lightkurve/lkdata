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
def dataframe(ntime, nrow, ncol):
    """DataFrame"""
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
def boolframe(ntime, nrow, ncol):
    """BoolFrame"""
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
def bitframe(ntime, nrow, ncol):
    """BitwiseFrame"""
    return BitwiseSeriesCollection(np.random.choice(64, (ntime, nrow * ncol)))


@pytest.fixture
def bitseries(ntime):
    """BitwiseSeries"""
    return BitwiseSeries(np.random.choice(64, (ntime)))


@pytest.fixture
def data_only(datacube, dataframe, dataseries):
    """DataSet with only data products, no error"""
    data = {
        "datacube": datacube,
        "dataframe": dataframe,
        "dataseries": dataseries,
    }
    return DataSet(data_products=data)


@pytest.fixture
def sample_dataset(
    datacube,
    dataframe,
    dataseries,
    boolcube,
    boolframe,
    boolseries,
    bitcube,
    bitframe,
    bitseries,
):
    """DataSet with all product types"""
    data = {
        "datacube": datacube,
        "dataframe": dataframe,
        "dataseries": dataseries,
    }
    bools = {
        "boolcube": boolcube,
        "boolframe": boolframe,
        "boolseries": boolseries,
    }
    codes = {
        "bitcube": bitcube,
        "bitframe": bitframe,
        "bitseries": bitseries,
    }
    return DataSet(data_products=data, bool_products=bools, bitwise_products=codes)


def test_dataset_init(data_only, sample_dataset, ntime):
    """Test init, minimal input"""
    # Data only set, empty bool and bitwise products
    assert isinstance(data_only.data_products, DataProducts)
    assert len(data_only) == ntime
    for key in ["datacube", "dataframe", "dataseries"]:
        assert key in data_only.data_products, f"{key} not in DataProducts"
    assert isinstance(data_only.bool_products, BoolProducts)
    assert not data_only.bool_products, "BoolProducts not empty"
    assert isinstance(data_only.bitwise_products, BitwiseProducts)
    assert not data_only.bitwise_products, "BitwiseProducts not empty"

    # DataSet with all the fixins'
    assert isinstance(sample_dataset.data_products, DataProducts)
    for key in ["datacube", "dataframe", "dataseries"]:
        assert key in sample_dataset.data_products, f"{key} not in DataProducts"
    assert isinstance(sample_dataset.bool_products, BoolProducts)
    for key in ["boolcube", "boolframe", "boolseries"]:
        assert key in sample_dataset.bool_products, f"{key} not in BoolProducts"
    assert isinstance(sample_dataset.bitwise_products, BitwiseProducts)
    for key in ["bitcube", "bitframe", "bitseries"]:
        assert key in sample_dataset.bitwise_products, f"{key} not in BitwiseProducts"


def test_dataset_getitem_string(data_only):
    """Test keyword retreival"""
    assert isinstance(data_only["datacube"], DataCube)
    assert isinstance(data_only["dataframe"], DataSeriesCollection)
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
    assert "Bool Products:" in repr_str
    assert "Bitwise Products:" in repr_str


def test_dataset_cubes_property(data_only):
    cubes = data_only.cubes
    assert "datacube" in cubes
    assert isinstance(cubes["datacube"], DataCube)


def test_dataset_frames_property(data_only):
    frames = data_only.frames
    assert "dataframe" in frames
    assert isinstance(frames["dataframe"], DataSeriesCollection)


def test_dataset_series_property(data_only):
    series = data_only.series
    assert "dataseries" in series
    assert isinstance(series["dataseries"], DataSeries)


def test_dataset_contents_property(data_only):
    contents = data_only.contents
    assert "datacube" in contents
    assert "dataframe" in contents
    assert "dataseries" in contents


def test_dataset_downsample(data_only):
    downsampled = data_only.downsample(nframes=5)
    assert all(val.shape[0] == 20 for val in downsampled.data_products.values())


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
    assert ds.user_kwargs == {"custom_param": "test"}


# class TestDataProcessor(unittest.TestCase):
#     def setUp(self):
#         self.processor = DataProcessorMixin()
#         self.processor._type = "data"
#         self.processor.kwargs = {}

#     def test_process_input_list(self):
#         data = [[1, 2], [3, 4]]
#         result = self.processor.process_input(data)
#         self.assertIsInstance(result, DataFrame)

#     def test_process_input_numpy_array(self):
#         data = np.array([[1, 2], [3, 4]])
#         result = self.processor.process_input(data)
#         self.assertIsInstance(result, DataFrame)

#     def test_process_input_data_cube(self):
#         data = DataCube(np.array([[[1, 2], [3, 4]]]))
#         result = self.processor.process_input(data)
#         self.assertIsInstance(result, DataCube)

#     def test_process_input_invalid_type(self):
#         with self.assertRaises(TypeError):
#             self.processor.process_input("invalid")

#     def test_check_attrs(self):
#         data = DataCube(np.array([[[1, 2], [3, 4]]]))
#         self.processor.ntime = 1
#         self.processor.nrow = 2
#         self.processor.ncol = 2
#         self.processor._check_attrs(data)

#     def test_check_attrs_mismatch(self):
#         data = DataCube(np.array([[[1, 2], [3, 4]]]))
#         self.processor.ntime = 2
#         with self.assertRaises(ValueError):
#             self.processor._check_attrs(data)


# class TestDataProducts:
#     @pytest.fixture
#     def product_bundle(self):
#         return DataProducts()

#     def test_unpack_data_dict(self, product_bundle):
#         data = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
#         result = product_bundle._unpack_data(data)
#         assert result == data

#     def test_unpack_data_datacube(self, product_bundle):
#         data = DataCube(np.random.rand(10, 5, 5))
#         result = product_bundle._unpack_data(data)
#         assert "flux_data" in result
#         assert isinstance(result["flux_data"], DataCube)

#     def test_unpack_data_dataframe(self, product_bundle):
#         data = DataFrame(np.random.rand(10, 25))
#         result = product_bundle._unpack_data(data)
#         assert "flux_data" in result
#         assert isinstance(result["flux_data"], DataFrame)

#     def test_unpack_data_dataseries(self, product_bundle):
#         data = DataSeries(np.random.rand(10))
#         result = product_bundle._unpack_data(data)
#         assert "flux_data" in result
#         assert isinstance(result["flux_data"], DataSeries)

#     def test_unpack_data_list(self, product_bundle):
#         data = [1, 2, 3, 4, 5]
#         result = product_bundle._unpack_data(data)
#         assert "flux_data" in result
#         assert isinstance(result["flux_data"], np.ndarray)
#         np.testing.assert_array_equal(result["flux_data"], np.array(data))

#     def test_unpack_data_numpy_array(self, product_bundle):
#         data = np.array([1, 2, 3, 4, 5])
#         result = product_bundle._unpack_data(data)
#         assert "flux_data" in result
#         assert isinstance(result["flux_data"], np.ndarray)
#         np.testing.assert_array_equal(result["flux_data"], data)

#     def test_unpack_data_unsupported_type(self, product_bundle):
#         data = 42  # integer is not a supported type
#         with pytest.raises(ValueError, match="Unsupported data type <class 'int'>"):
#             product_bundle._unpack_data(data)


# class TestEmptyDataset:
#     @pytest.fixture
#     def sample_dataset(self):
#         # Empty datset, but has attributes to check against
#         ds = DataSet()
#         ds.ntime = 10
#         ds.nrow = 5
#         ds.ncol = 5
#         return ds

#     @pytest.fixture
#     def sample_datacube(self):
#         data = np.random.rand(10, 5, 5)
#         return DataCube(data)

#     @pytest.fixture
#     def sample_dataframe(self):
#         data = pd.DataFrame(np.random.rand(10, 25))
#         return DataFrame(data)

#     def test_empty(self):
#         ds = DataSet()
#         assert ds.index.empty
#         assert ds.ntime == 0

#     def test_check_attrs_matching(self, sample_dataset, sample_datacube):
#         sample_dataset.data._check_attrs(sample_datacube)
#         assert sample_dataset.ntime == sample_datacube.ntime
#         assert sample_dataset.data.nrow == sample_datacube.nrow
#         assert sample_dataset.data.ncol == sample_datacube.ncol

#     def test_check_attrs_mismatch(self, sample_dataset, sample_datacube):
#         sample_dataset.data.ntime = 11  # Mismatch
#         with pytest.raises(
#             ValueError, match="Dataset value for ntime != given data ntime"
#         ):
#             sample_dataset.data._check_attrs(sample_datacube)

#     def test_check_attrs_index_mismatch(self, sample_dataset, sample_datacube):
#         sample_dataset.ntime = 10
#         sample_dataset.nrow = 5
#         sample_dataset.ncol = 5
#         sample_dataset.data.index = pd.MultiIndex.from_product([range(11), ["A", "B"]])
#         with pytest.raises(
#             ValueError, match="Dataset shape for index does not match given data index"
#         ):
#             sample_dataset.data._check_attrs(sample_datacube)

#     def test_check_attrs_columns_mismatch(self, sample_dataset, sample_datacube):
#         sample_dataset.data.columns = pd.MultiIndex.from_product([range(6), ["X", "Y"]])
#         with pytest.raises(
#             ValueError,
#             match="Dataset shape for columns does not match given data columns",
#         ):
#             sample_dataset.data._check_attrs(sample_datacube)

#     def test_check_attrs_dataframe(self, sample_dataset, sample_dataframe):
#         sample_dataframe = DataFrame(np.random.rand(10, 25))
#         sample_dataset.ntime = 10
#         sample_dataset.data._check_attrs(sample_dataframe)
#         assert sample_dataset.ntime == sample_dataframe.ntime

#     def test_check_attrs_with_non_standard_attribute(
#         self, sample_dataset, sample_datacube
#     ):
#         sample_datacube.custom_attr = "test"
#         sample_dataset.data._check_attrs(sample_datacube)
#         assert not hasattr(sample_dataset, "custom_attr")

#     def test_dataset_user_kwargs(
#         self,
#     ):
#         ds = DataSet(user_param="test")
#         assert ds.user_kwargs == {"user_param": "test"}


# class TestDatasetFunctions:
#     @pytest.fixture
#     def time(self):
#         # TODO: Support datetime objects
#         # time = pd.date_range(start='2020-01-01', periods=100, freq='h')
#         return np.linspace(0, 4 * np.pi, 100)

#     @pytest.fixture
#     def sample_data(self, time):
#         data = {
#             "cube": DataCube(np.random.rand(100, 5, 3)),
#             "frame": DataFrame(np.random.rand(100, 5)),
#             "series": DataSeries(np.random.rand(100)),
#         }
#         # error = {
#         #     "errcube": ErrorCube(np.random.rand(100, 5, 3)),
#         #     "errframe": ErrorFrame(np.random.rand(100, 5)),
#         #     "errseries": ErrorSeries(np.random.rand(100)),
#         # }
#         return DataSet(data, time_indices={"time": time})

#     def test_dataset_fold(self, sample_data):
#         folded = sample_data.fold(period=24)
#         assert "phase" in folded.index.names
#         assert np.all(folded.index.get_level_values("phase") < 1)

#     def test_dataset_fold_inplace(self, sample_data):
#         sample_data.fold(period=24, inplace=True)
#         assert "phase" in sample_data.index.names
#         assert np.all(sample_data.index.get_level_values("phase") < 1)

#     def test_dataset_fold_with_t0(self, sample_data):
#         folded = sample_data.fold(period=24, t0=10)
#         assert "phase" in folded.index.names
#         assert np.all(folded.index.get_level_values("phase") < 1)
#         assert np.all(folded.index.get_level_values("phase") >= 0)

#     def test_dataset_fold_existing_phase(self, sample_data):
#         folded = sample_data.fold(period=24)
#         folded = folded.fold(period=24)
#         assert "phase" in folded.index.names
#         assert np.all(folded.index.get_level_values("phase") < 1)
#         assert np.all(folded.index.get_level_values("phase") >= 0)

#     def test_dataset_fold_level_selection(self, time):
#         data = {
#             "series": DataSeries(
#                 np.sin(np.linspace(0, 4 * np.pi, 100)),
#                 index=pd.MultiIndex.from_arrays(
#                     [range(100), time], names=["cadence", "time"]
#                 ),
#             )
#         }
#         ds = DataSet(data)
#         folded = ds.fold(period=24, level="time")
#         assert "phase" in folded.index.names
#         assert np.all(folded.index.get_level_values("phase") < 1)
#         assert "cadence" in folded.index.names

#     def test_dataset_fold_label(self, time):
#         data = {
#             "series": DataSeries(np.sin(np.linspace(0, 4 * np.pi, 100)), index=time)
#         }
#         ds = DataSet(data)
#         folded = ds.fold(period=24, label="custom_phase")
#         assert "custom_phase" in folded.index.names

#     def test_dataset_fold_with_negative_period(self, time):
#         data = {"series": DataSeries(np.random.rand(100), index=time)}
#         ds = DataSet(data)
#         with pytest.raises(ValueError):
#             ds.fold(period=-24)

#     def test_dataset_fold_with_multiple_periods(self, time):
#         data = {
#             "series": DataSeries(np.sin(np.linspace(0, 4 * np.pi, 100)), index=time)
#         }
#         ds = DataSet(data)
#         folded1 = ds.fold(period=24)
#         folded2 = ds.fold(period=12)
#         assert any(
#             folded1.index.get_level_values("phase")
#             != folded2.index.get_level_values("phase")
#         )

#     def test_dataset_downsample(self, sample_data):
#         downsampled = sample_data.downsample(nframes=10, level=1)

#         assert all(val.shape[0] == 10 for val in downsampled.data.values())

#     # def test_dataset_downsample_with_non_integer_nframes(self, sample_data):
#     #     with pytest.raises(ValueError):
#     #         sample_data.downsample(nframes=3.5)

#     def test_dataset_droplevel(self, sample_data):
#         folded = sample_data.fold(period=0.4)
#         dropped = folded.droplevel(level="phase")
#         assert all(val.index.nlevels == 2 for val in dropped.data.values())
#         assert all(val.index.nlevels == 2 for val in dropped.error.values())


# class TestDataSet1(unittest.TestCase):
#     def setUp(self):
#         self.data = np.array([[1, 2], [3, 4]])
#         self.error = np.array([[0.1, 0.2], [0.3, 0.4]])

#     def test_init(self):
#         ds = DataSet(self.data, self.error)
#         self.assertIsInstance(ds.data, DataProducts)

#     def test_cubes_property(self):
#         cube_data = np.array([[[1, 2], [3, 4]]])
#         cube_error = np.array([[[0.1, 0.2], [0.3, 0.4]]])
#         ds = DataSet({"data_cube": DataCube(cube_data, uncertainty=cube_error)})
#         cubes = ds.cubes
#         self.assertIn("data_cube", cubes)
#         self.assertIsInstance(cubes["data_cube"], DataCube)

#     def test_frames_property(self):
#         ds = DataSet(
#             {"data_frame": DataFrame(self.data)},
#         )
#         frames = ds.frames
#         self.assertIn("data_frame", frames)
#         self.assertIn("error_frame", frames)
#         self.assertIsInstance(frames["data_frame"], DataFrame)

#     def test_series_property(self):
#         series_data = np.array([1, 2, 3, 4])
#         series_error = np.array([0.1, 0.2, 0.3, 0.4])
#         ds = DataSet(
#             {"data_series": DataSeries(series_data, uncertainty=series_error)},
#         )
#         series = ds.series
#         self.assertIn("data_series", series)
#         self.assertIn("error_series", series)
#         self.assertIsInstance(series["data_series"], DataSeries)

#     def test_getitem_string(self):
#         ds = DataSet({"data": self.data}, {"error": self.error})
#         self.assertIsInstance(ds["data"], DataFrame)

#     def test_getitem_slice(self):
#         ds = DataSet({"data": self.data}, {"error": self.error})
#         sliced = ds[0:1]
#         self.assertIsInstance(sliced, DataSet)
#         np.testing.assert_array_equal(sliced.data["data"], self.data[0:1])
#         np.testing.assert_array_equal(sliced.error["error"], self.error[0:1])

#     def test_getitem_invalid_key(self):
#         ds = DataSet({"data": self.data}, {"error": self.error})
#         with self.assertRaises(ValueError):
#             ds["invalid_key"]

#     def test_repr(self):
#         ds = DataSet({"data": self.data}, {"error": self.error})
#         repr_str = repr(ds)
#         self.assertIn("data", repr_str)
#         self.assertIn("error", repr_str)


# class TestDataSet2:
#     @pytest.fixture
#     def sample_data(self):
#         time = np.linspace(0, 4 * np.pi, 10)
#         data = {
#             "cube": DataCube(np.random.rand(10, 5, 3)),
#             "frame": DataFrame(np.random.rand(10, 5)),
#             "series": DataSeries(np.random.rand(10)),
#         }
#         # error = {
#         #     "errcube": ErrorCube(np.random.rand(10, 5, 3)),
#         #     "errframe": ErrorFrame(np.random.rand(10, 5)),
#         #     "errseries": ErrorSeries(np.random.rand(10)),
#         # }
#         return DataSet(data, time_indices={"time": time})

#     def test_dataset_ntime(self, sample_data):
#         assert sample_data.ntime == 10
#         assert len(sample_data.index) == 10

#         # Test setter
#         sample_data.ntime = 15
#         assert sample_data.data.ntime == 15
#         assert sample_data.error.ntime == 15
#         assert len(sample_data.index) == 10

#     def test_dataset_index(self, sample_data):
#         for val in sample_data.data.values():
#             assert (sample_data.index == val.index).all()
#         for val in sample_data.error.values():
#             assert (sample_data.index == val.index).all()

#         # Test setter
#         new_time = np.linspace(0, 4 * np.pi, 10)
#         new_index = DataSet(time_indices={"time": new_time}).index
#         sample_data.index = new_index
#         for val in sample_data.data.values():
#             assert (val.index == new_index).all()
#         for val in sample_data.error.values():
#             assert (val.index == new_index).all()

#     def test_dataset_repr(self, sample_data):
#         repr_str = repr(sample_data)
#         assert "Data Products:" in repr_str
#         assert "Error Products:" in repr_str
#         assert "Properties:" in repr_str

#     def test_dataset_getitem_invalid_key(self, sample_data):
#         with pytest.raises(ValueError):
#             _ = sample_data["invalid_key"]

#     def test_dataset_slice(self, sample_data):
#         sliced = sample_data[1:5]
#         assert all(val.shape[0] == 4 for val in sliced.data.values())
#         assert all(val.shape[0] == 4 for val in sliced.error.values())

#     def test_dataset_getitem_tuple(self, sample_data):
#         subset = sample_data[1:5, :]
#         assert all(
#             isinstance(val, (DataCube, DataFrame, DataSeries))
#             for val in subset.data.values()
#         )

#     def test_dataset_getitem_tuple_invalid(self):
#         ds = DataSet()
#         with pytest.raises(KeyError):
#             _ = ds[1, 2, 3, 4]

#     def test_dataset_build_instance(self, sample_data):
#         data = sample_data.data
#         ds = DataSet(data, custom_param="test")
#         new_ds = ds._build_instance(data, new_param="new_test")
#         assert isinstance(new_ds, DataSet)
#         assert new_ds.custom_param == "test"
#         assert new_ds.new_param == "new_test"

#     def test_dataset_getitem_single_element(self, sample_data):
#         single_element = sample_data[0]
#         assert isinstance(single_element, DataSet)
#         assert all(val.shape[0] == 1 for val in single_element.data.values())
#         assert all(val.shape[0] == 1 for val in single_element.error.values())

#     def test_dataset_getitem_tuple_single_element(self, sample_data):
#         single_element = sample_data[0, :]
#         assert isinstance(single_element, DataSet)
#         assert all(val.shape[0] == 1 for val in single_element.data.values())
#         assert all(val.shape[0] == 1 for val in single_element.error.values())

#     def test_dataset_getitem_tuple_column_slice(self, sample_data):
#         column_slice = sample_data[:, 1:3]
#         assert isinstance(column_slice, DataSet)
#         assert all(sample_data["cube"][:, 1:3] == column_slice["cube"])
#         assert all(sample_data["errcube"][:, 1:3] == column_slice["errcube"])

#     def test_dataset_getitem_tuple_3d_slice(self, sample_data):
#         slice_3d = sample_data[:, 1:3, 0]
#         assert isinstance(slice_3d, DataSet)
#         assert all(
#             val.shape == (10, 2)
#             for val in slice_3d.data.values()
#             if isinstance(val, DataCube)
#         )

#     def test_dataset_user_kwargs_preservation(self):
#         ds = DataSet(custom_param1="test1", custom_param2="test2")
#         assert "custom_param1" in ds._user_kwargs
#         assert "custom_param2" in ds._user_kwargs
#         assert ds.custom_param1 == "test1"
#         assert ds.custom_param2 == "test2"

#         assert "ntime" not in ds._user_kwargs
#         assert "index" not in ds._user_kwargs
#         assert "columns" not in ds._user_kwargs

#     def test_dataset_with_pandas_dataframe(
#         self,
#     ):
#         df = pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"])
#         ds = DataSet(data={"df": df})
#         assert isinstance(ds.data["df"], DataFrame)

#     def test_dataset_with_numpy_array(
#         self,
#     ):
#         arr = np.random.rand(10, 3, 2)
#         ds = DataSet(data={"arr": arr})
#         assert isinstance(ds.data["arr"], DataCube)

#     def test_dataset_with_mixed_data_types(
#         self,
#     ):
#         data = {
#             "arr": np.random.rand(10, 3, 2),
#             "df": pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"]),
#             "series": pd.Series(np.random.rand(10)),
#         }
#         ds = DataSet(data=data)
#         assert isinstance(ds.data["arr"], DataCube)
#         assert isinstance(ds.data["df"], DataFrame)
#         assert isinstance(ds.data["series"], DataSeries)

#         data = {
#             "cube": DataCube(np.random.rand(10, 3, 2)),
#             "frame": DataFrame(np.random.rand(10, 3)),
#             "series": DataSeries(np.random.rand(10)),
#         }
#         # error = {
#         #     "cube": ErrorCube(np.random.rand(10, 3, 2)),
#         #     "frame": ErrorFrame(np.random.rand(10, 3)),
#         #     "series": ErrorSeries(np.random.rand(10)),
#         # }
#         ds = DataSet(data=data)
#         assert isinstance(ds.data["cube"], DataCube)
#         assert isinstance(ds.data["frame"], DataFrame)
#         assert isinstance(ds.data["series"], DataSeries)

#     def test_dataset_getitem_with_boolean_indexing(self):
#         data = {"series": DataSeries(np.random.rand(10))}
#         ds = DataSet(data)
#         mask = np.array([True, False] * 5)
#         subset = ds[list(mask)]
#         assert subset.data["series"].shape[0] == 5

#     def test_dataset_getitem_with_complex_boolean_indexing(self):
#         data = {"frame": DataFrame(np.random.rand(10, 3))}
#         ds = DataSet(data)
#         mask = (ds.data["frame"].iloc[:, 0] > 0.5) & (ds.data["frame"].iloc[:, 1] < 0.7)
#         subset = ds[list(mask)]
#         assert subset.data["frame"].shape[0] == mask.sum()

#     def test_dataset_with_inconsistent_index(self):
#         data = {
#             "series1": DataSeries(np.random.rand(10), index=pd.RangeIndex(0, 10)),
#             "series2": DataSeries(np.random.rand(15), index=pd.RangeIndex(10, 25)),
#         }
#         with pytest.raises(ValueError):
#             DataSet(data)
